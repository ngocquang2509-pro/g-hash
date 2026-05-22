import torch
import torch.nn as nn
import timm


class VisionEncoder(nn.Module):
    """
    Vision Transformer encoder for image feature extraction
    Uses pretrained ViT models from timm library
    """
    
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, feature_dim=768):
        super(VisionEncoder, self).__init__()
        
        # Load pretrained ViT model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # We'll use CLS token
        )
        self.feature_dim = feature_dim
        
        # Get actual output dimension from model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = self.backbone(dummy_input)
            if len(output.shape) == 3:  # (B, N_patches, D)
                self.backbone_out_dim = output.shape[-1]
            else:
                self.backbone_out_dim = output.shape[-1]
        
    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            features: Global image features (B, feature_dim)
        """
        # Extract features using ViT
        features = self.backbone(x)
        
        # If output is sequence (B, N_patches, D), take CLS token (first token)
        if len(features.shape) == 3:
            features = features[:, 0]  # CLS token
        
        return features
    
    @property
    def output_dim(self):
        return self.backbone_out_dim


class TextEncoder(nn.Module):
    """
    Simple text encoder for label embeddings
    Can use pretrained word embeddings (GloVe, Word2Vec) or learnable embeddings
    """
    
    def __init__(self, num_classes, embed_dim=300, hidden_dim=512):
        super(TextEncoder, self).__init__()
        
        # Learnable label embeddings
        self.embeddings = nn.Embedding(num_classes, embed_dim)
        
        # Optional MLP for projection
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, label_indices=None):
        """
        Args:
            label_indices: Label indices (optional, default to all labels)
        
        Returns:
            embeddings: Label embeddings (N_labels, hidden_dim)
        """
        if label_indices is None:
            # Return all label embeddings
            label_indices = torch.arange(self.embeddings.num_embeddings, 
                                        device=self.embeddings.weight.device)
        
        # Get embeddings
        embeds = self.embeddings(label_indices)
        
        # Project to hidden space
        embeds = self.projection(embeds)
        
        return embeds
    
    def load_pretrained_embeddings(self, pretrained_weights):
        """Load pretrained word embeddings (e.g., GloVe)"""
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_weights))
