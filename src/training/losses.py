import torch
import torch.nn as nn
import torch.nn.functional as F


class GHashLoss(nn.Module):
    """
    Combined loss function for G-hash model
    
    L_total = L_cls + α * L_sim + β * L_quant
    
    - L_cls: Classification loss (BCE for multi-label)
    - L_sim: Similarity preservation loss (contrastive/triplet)
    - L_quant: Quantization loss (minimize difference between continuous and binary)
    """
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=1.0, delta=0.5,
                 eta=0.5, pos_weight=None):
        super(GHashLoss, self).__init__()
        self.alpha = alpha  # Weight for similarity loss
        self.beta = beta    # Weight for quantization loss
        self.gamma = gamma  # Weight for classification loss
        self.delta = delta  # Weight for bit balance loss
        self.eta = eta      # Weight for image-image retrieval loss

        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight.float())
        else:
            self.pos_weight = None

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
    def forward(self, img_hash, txt_hash, pred_labels, true_labels):
        """
        Args:
            img_hash: Image hash codes (B, hash_bits) - continuous values from tanh
            txt_hash: Text hash codes (num_classes, hash_bits) - continuous values
            pred_labels: Predicted label logits (B, num_classes)
            true_labels: Ground truth labels (B, num_classes) - multi-hot vectors
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses for logging
        """
        # 1. Classification Loss
        loss_cls = self.bce_loss(pred_labels, true_labels)
        
        # 2. Similarity Loss (Weighted Cosine Similarity)
        # For each image, compute similarity with its corresponding label hash codes
        # Positive pairs: image and its true labels should have similar hash codes
        # Negative pairs: image and other labels should have dissimilar hash codes
        
        loss_sim = self.similarity_loss(img_hash, txt_hash, true_labels)
        loss_retrieval = self.image_retrieval_loss(img_hash, true_labels)
        
        # 3. Quantization Loss
        # Minimize ||tanh(h) - sign(tanh(h))||^2
        # This encourages hash codes to be close to {-1, +1}
        loss_quant_img = self.quantization_loss(img_hash)
        loss_quant_txt = self.quantization_loss(txt_hash)
        loss_quant = loss_quant_img + loss_quant_txt
        
        # 4. Orthogonality Loss (Chống Sập Mode / Mode Collapse)
        # Bắt buộc các Text Hash Code của các Nhãn KHÁC NHAU (Ví dụ: Ngồi vs Đứng) 
        # PHẢI đẩy nhau ra xa (Trực giao) thay vì bị GAT làm mờ nhòe dính vào làm 1.
        txt_hash_norm = F.normalize(txt_hash, p=2, dim=1)
        txt_sim = torch.matmul(txt_hash_norm, txt_hash_norm.t())
        eye = torch.eye(txt_hash.size(0), device=txt_hash.device)
        loss_ortho = F.mse_loss(txt_sim, eye) * 2.0  # Phạt nặng việc dính chùm
        
        # 5. Bit Balance Loss (Cân bằng Bit cho Ảnh để chống Mode Collapse)
        # Ép phân phối bit ngang ngửa 50/50, cấm ảnh nào cũng toàn 111 hoặc 000
        loss_bit_balance = torch.mean(img_hash.mean(dim=0) ** 2)
        
        # Total loss
        total_loss = (self.gamma * loss_cls + 
                     self.alpha * loss_sim + 
                     self.eta * loss_retrieval +
                     self.beta * loss_quant +
                     loss_ortho +
                     self.delta * loss_bit_balance)
        
        # Return individual losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'classification': loss_cls.item(),
            'similarity': loss_sim.item(),
            'retrieval': loss_retrieval.item(),
            'quantization': loss_quant.item(),
            'orthogonality': loss_ortho.item(),
            'bit_balance': loss_bit_balance.item()
        }
        
        return total_loss, loss_dict
    
    def similarity_loss(self, img_hash, txt_hash, labels):
        """
        Similarity preservation loss using cosine similarity
        
        Args:
            img_hash: (B, hash_bits)
            txt_hash: (num_classes, hash_bits)
            labels: (B, num_classes) multi-hot vectors
        """
        batch_size = img_hash.size(0)
        num_classes = txt_hash.size(0)
        
        # Normalize hash codes
        img_hash_norm = F.normalize(img_hash, p=2, dim=1)  # (B, hash_bits)
        txt_hash_norm = F.normalize(txt_hash, p=2, dim=1)  # (num_classes, hash_bits)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(img_hash_norm, txt_hash_norm.t())  # (B, num_classes)
        
        # Create similarity targets based on labels
        # If image has label k, similarity should be high; otherwise low
        # Normalize labels to get weights
        label_weights = labels / (labels.sum(dim=1, keepdim=True) + 1e-8)
        
        # Positive samples: maximize similarity for true labels
        pos_sim = (similarity_matrix * labels).sum() / (labels.sum() + 1e-8)
        
        # Negative samples: minimize similarity for false labels
        neg_labels = 1 - labels
        neg_sim = (similarity_matrix * neg_labels).sum() / (neg_labels.sum() + 1e-8)
        
        # Loss: encourage positive similarity, discourage negative similarity
        # Using margin-based loss
        margin = 0.5
        loss = torch.clamp(margin + neg_sim - pos_sim, min=0)
        
        return loss
    
    def quantization_loss(self, hash_codes):
        """
        Quantization loss to encourage binary values
        
        Args:
            hash_codes: Continuous hash codes from tanh (values in [-1, 1])
        
        Returns:
            Quantization loss
        """
        # Encourage hash codes to be close to -1 or +1
        # ||h - sign(h)||^2 = sum((h - sign(h))^2)
        binary_codes = torch.sign(hash_codes)
        binary_codes[binary_codes == 0] = 1  # Handle zeros
        
        loss = F.mse_loss(hash_codes, binary_codes.detach())
        return loss

    def image_retrieval_loss(self, img_hash, labels):
        """
        Supervised pairwise loss for image-image retrieval.

        Images that share at least one label are treated as positives.
        """
        if img_hash.size(0) < 2:
            return img_hash.new_tensor(0.0)

        similarity_targets = (labels @ labels.t()) > 0
        diagonal_mask = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)

        pos_mask = similarity_targets & (~diagonal_mask)
        neg_mask = (~similarity_targets) & (~diagonal_mask)

        sim_logits = torch.matmul(img_hash, img_hash.t()) / img_hash.size(1)

        losses = []
        if pos_mask.any():
            losses.append(F.softplus(-sim_logits[pos_mask]).mean())
        if neg_mask.any():
            losses.append(F.softplus(sim_logits[neg_mask]).mean())

        if not losses:
            return img_hash.new_tensor(0.0)

        return sum(losses) / len(losses)


class ContrastiveLoss(nn.Module):
    """
    Alternative: Contrastive loss for hash code learning
    """
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, hash1, hash2, label):
        """
        Args:
            hash1, hash2: Hash codes of two samples
            label: 1 if similar, 0 if dissimilar
        """
        euclidean_distance = F.pairwise_distance(hash1, hash2)
        
        loss = (label * torch.pow(euclidean_distance, 2) +
                (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss.mean()


class TripletLoss(nn.Module):
    """
    Alternative: Triplet loss for hash learning
    """
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor hash codes
            positive: Positive hash codes (similar to anchor)
            negative: Negative hash codes (dissimilar to anchor)
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean()
