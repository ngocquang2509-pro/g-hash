import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat import GAT
from .vision_encoder import VisionEncoder, TextEncoder


class GHashModel(nn.Module):
    """
    G-hash: Educational Image Retrieval based on GAT integrated with Deep Hashing
    
    Architecture:
        1. Image Stream: Vision Transformer -> Hash Code
        2. Label Stream: Text Embeddings -> GAT -> Enhanced Label Features
        3. Hashing: tanh activation for continuous codes, sign for binary
    """
    
    def __init__(self, config):
        super(GHashModel, self).__init__()
        
        # Configuration
        self.num_classes = config['dataset']['num_classes']
        self.hash_bits = config['model']['hash_bits']
        self.hidden_dim = config['model']['hidden_dim']
        
        # Image encoder (Vision Transformer)
        self.image_encoder = VisionEncoder(
            model_name=config['model']['image_encoder'],
            pretrained=True
        )
        img_feat_dim = self.image_encoder.output_dim
        
        # Text encoder (Label embeddings)
        self.text_encoder = TextEncoder(
            num_classes=self.num_classes,
            embed_dim=config['model']['text_embed_dim'],
            hidden_dim=self.hidden_dim
        )
        
        # GAT for label correlation learning
        self.use_gat = config.get('gat', {}).get('use_gat', True)
        if self.use_gat:
            self.gat = GAT(
                in_features=self.hidden_dim,
                hidden_features=config['gat']['hidden_dim'],
                out_features=self.hidden_dim,
                num_heads=config['gat']['num_heads'],
                num_layers=config['gat']['num_layers'],
                dropout=config['gat']['dropout']
            )
        else:
            self.gat = None
        
        # Hash layers (Dùng GELU và LayerNorm để chống Sốc Cấp Cứu cho Nơ-ron)
        self.img_hash_fc = nn.Sequential(
            nn.LayerNorm(img_feat_dim),          # Cân bằng huyết áp tín hiệu từ ViT
            nn.Linear(img_feat_dim, self.hidden_dim),
            nn.GELU(),                           # Mượt mà hơn, triệt tiêu Chết nơ-ron
            nn.Dropout(config['model']['dropout']),
            nn.Linear(self.hidden_dim, self.hash_bits)
        )
        
        self.txt_hash_fc = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hash_bits)
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(img_feat_dim),
            nn.Linear(img_feat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        
    def forward(self, images, labels, adj_matrix, return_features=False):
        """
        Forward pass through G-hash model
        
        Args:
            images: Input images (B, 3, H, W)
            labels: Multi-hot label vectors (B, num_classes)
            adj_matrix: Label co-occurrence adjacency matrix (num_classes, num_classes)
            return_features: Whether to return intermediate features
        
        Returns:
            img_hash: Image hash codes (B, hash_bits)
            txt_hash: Text hash codes (num_classes, hash_bits)
            pred_labels: Predicted label probabilities (B, num_classes)
        """
        batch_size = images.size(0)
        
        # ===== Image Path =====
        # Extract image features using ViT
        img_features = self.image_encoder(images)  # (B, img_feat_dim)
        
        # Generate image hash codes
        img_hash = torch.tanh(self.img_hash_fc(img_features))  # (B, hash_bits)
        
        # ===== Label Path =====
        # Get all label embeddings
        label_embeddings = self.text_encoder()  # (num_classes, hidden_dim)
        
        # Apply GAT to learn label correlations
        if self.use_gat:
            enhanced_labels = self.gat(label_embeddings, adj_matrix)  # (num_classes, hidden_dim)
        else:
            enhanced_labels = label_embeddings
        
        # Generate text hash codes for all labels
        txt_hash = torch.tanh(self.txt_hash_fc(enhanced_labels))  # (num_classes, hash_bits)
        
        # ===== Label Classification =====
        # Use enhanced label features for classification
        # We need to aggregate label features for each image based on its labels
        # Alternative: use image features for classification
        pred_labels = self.classifier(img_features)  # (B, num_classes)
        
        if return_features:
            return {
                'img_hash': img_hash,
                'txt_hash': txt_hash,
                'pred_labels': pred_labels,
                'img_features': img_features,
                'enhanced_labels': enhanced_labels
            }
        
        return img_hash, txt_hash, pred_labels
    
    def generate_hash_code(self, images):
        """
        Generate binary hash codes for images (inference mode)
        
        Args:
            images: Input images (B, 3, H, W)
        
        Returns:
            Binary hash codes (B, hash_bits) in {-1, +1}
        """
        self.eval()
        with torch.no_grad():
            img_features = self.image_encoder(images)
            continuous_hash = self.img_hash_fc(img_features)
            binary_hash = torch.sign(continuous_hash)
            # Handle zero values (sign(0) = 0, we want -1 or +1)
            binary_hash[binary_hash == 0] = 1
        return binary_hash
    
    def get_label_hash_codes(self, adj_matrix):
        """
        Generate binary hash codes for all labels
        
        Args:
            adj_matrix: Label co-occurrence adjacency matrix
        
        Returns:
            Binary hash codes for labels (num_classes, hash_bits)
        """
        self.eval()
        with torch.no_grad():
            label_embeddings = self.text_encoder()
            if self.use_gat:
                enhanced_labels = self.gat(label_embeddings, adj_matrix)
            else:
                enhanced_labels = label_embeddings
            continuous_hash = self.txt_hash_fc(enhanced_labels)
            binary_hash = torch.sign(continuous_hash)
            binary_hash[binary_hash == 0] = 1
        return binary_hash


class BaselineModel(nn.Module):
    """
    Baseline model without GAT for comparison
    Uses ResNet or simple CNN instead of ViT
    """
    
    def __init__(self, config):
        super(BaselineModel, self).__init__()
        
        self.num_classes = config['dataset']['num_classes']
        self.hash_bits = config['model']['hash_bits']
        self.hidden_dim = config['model']['hidden_dim']
        
        # Use ResNet instead of ViT
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)
        # Remove final FC layer
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        img_feat_dim = 2048  # ResNet50 output dim
        
        # Direct hash projection (no GAT)
        self.hash_fc = nn.Sequential(
            nn.Linear(img_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hash_bits)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(img_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        
    def forward(self, images, labels=None):
        # Extract features
        features = self.image_encoder(images)
        features = features.view(features.size(0), -1)
        
        # Generate hash codes
        hash_code = torch.tanh(self.hash_fc(features))
        
        # Predict labels
        pred_labels = self.classifier(features)
        
        return hash_code, None, pred_labels
    
    def generate_hash_code(self, images):
        self.eval()
        with torch.no_grad():
            features = self.image_encoder(images)
            features = features.view(features.size(0), -1)
            continuous_hash = self.hash_fc(features)
            binary_hash = torch.sign(continuous_hash)
            binary_hash[binary_hash == 0] = 1
        return binary_hash
