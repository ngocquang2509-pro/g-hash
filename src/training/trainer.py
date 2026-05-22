import copy
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import time

from ..evaluation.metrics import compute_retrieval_metrics
from ..data.label_graph import build_label_cooccurrence_matrix


class Trainer:
    """
    Trainer for G-hash model
    """
    
    def __init__(self, model, train_loader, test_loader, query_loader, 
                 criterion, config, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.query_loader = query_loader
        self.criterion = criterion
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs']
        )
        
        # Build label co-occurrence graph
        self.adj_matrix = self._build_adjacency_matrix()
        
        # Training history
        self.history = {
            'train_loss': [],
            'loss_components': {
                'total': [],
                'classification': [],
                'similarity': [],
                'retrieval': [],
                'quantization': [],
                'orthogonality': [],
                'bit_balance': []
            },
            'eval_metrics': []
        }
        
        # Early stopping
        self.best_map = 0
        self.patience_counter = 0
        self.patience = config['training']['early_stopping_patience']
        
        # Save directory
        self.save_dir = Path(config['save_dir']) / time.strftime("%Y%m%d-%H%M%S")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.database_eval_loader = self._create_database_eval_loader()
        
    def _build_adjacency_matrix(self):
        """Build label co-occurrence adjacency matrix from training data"""
        print("Building label co-occurrence graph...")
        
        all_labels = []
        for _, labels, _ in self.train_loader:
            all_labels.append(labels)
        
        all_labels = torch.cat(all_labels, dim=0)
        adj_matrix = build_label_cooccurrence_matrix(all_labels)
        
        print(f"Label graph built: {adj_matrix.shape}")
        return adj_matrix.to(self.device)

    def _create_database_eval_loader(self):
        """Create a deterministic database loader for retrieval evaluation."""
        if self.config.get('evaluation', {}).get('database_split', 'train') != 'train':
            return self.test_loader

        dataset = copy.copy(self.train_loader.dataset)
        dataset.transform = getattr(self.query_loader.dataset, 'transform', dataset.transform)

        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 0)
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0
        epoch_loss_components = {
            'classification': 0,
            'similarity': 0,
            'retrieval': 0,
            'quantization': 0,
            'orthogonality': 0,
            'bit_balance': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            img_hash, txt_hash, pred_labels = self.model(images, labels, self.adj_matrix)
            
            # Compute loss
            loss, loss_dict = self.criterion(img_hash, txt_hash, pred_labels, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            for key in epoch_loss_components:
                epoch_loss_components[key] += loss_dict[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'cls': loss_dict['classification'],
                'sim': loss_dict['similarity'],
                'ret': loss_dict['retrieval'],
                'quan': loss_dict['quantization']
            })
        
        # Average losses
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        for key in epoch_loss_components:
            epoch_loss_components[key] /= num_batches
        
        return epoch_loss, epoch_loss_components
    
    def evaluate(self, epoch=0):
        """Evaluate model on test set"""
        self.model.eval()
        
        print(f"\nEvaluating epoch {epoch}...")
        database_split = self.config.get('evaluation', {}).get('database_split', 'train')
        
        with torch.no_grad():
            # Generate hash codes for database
            db_codes = []
            db_labels = []

            database_loader = self.database_eval_loader if database_split == 'train' else self.test_loader
            for images, labels, _ in tqdm(database_loader, desc="Database encoding"):
                images = images.to(self.device)
                codes = self.model.generate_hash_code(images)
                db_codes.append(codes.cpu())
                db_labels.append(labels)
            
            db_codes = torch.cat(db_codes, dim=0).numpy()
            db_labels = torch.cat(db_labels, dim=0).numpy()
            
            # Generate hash codes for queries
            query_codes = []
            query_labels = []
            
            for images, labels, _ in tqdm(self.query_loader, desc="Query encoding"):
                images = images.to(self.device)
                codes = self.model.generate_hash_code(images)
                query_codes.append(codes.cpu())
                query_labels.append(labels)
            
            query_codes = torch.cat(query_codes, dim=0).numpy()
            query_labels = torch.cat(query_labels, dim=0).numpy()
        
        # Compute retrieval metrics
        metrics = compute_retrieval_metrics(
            query_codes, db_codes, query_labels, db_labels,
            top_k_list=self.config['evaluation']['top_k']
        )
        
        # Print metrics
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS - Epoch {epoch}")
        print("="*60)
        for key, value in metrics.items():
            if not isinstance(value, dict):
                print(f"{key:20s}: {value:.4f}")
        print("="*60 + "\n")
        
        return metrics
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Model: G-hash")
        print(f"Device: {self.device}")
        print(f"Hash bits: {self.config['model']['hash_bits']}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Query samples: {len(self.query_loader.dataset)}")
        print(f"Save directory: {self.save_dir}")
        print("="*60 + "\n")
        
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train one epoch
            train_loss, loss_components = self.train_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log training loss
            self.history['train_loss'].append(train_loss)
            self.history['loss_components']['total'].append(train_loss)
            for key, value in loss_components.items():
                self.history['loss_components'][key].append(value)
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"  - Classification: {loss_components['classification']:.4f}")
            print(f"  - Similarity: {loss_components['similarity']:.4f}")
            print(f"  - Retrieval: {loss_components['retrieval']:.4f}")
            print(f"  - Quantization: {loss_components['quantization']:.4f}")
            
            # Evaluate every few epochs
            if epoch % 5 == 0 or epoch == num_epochs:
                metrics = self.evaluate(epoch)
                self.history['eval_metrics'].append({
                    'epoch': epoch,
                    'metrics': metrics
                })
                
                # Check for improvement
                current_map = metrics['mAP']
                if current_map > self.best_map:
                    self.best_map = current_map
                    self.patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"✓ New best mAP: {self.best_map:.4f} - Model saved!")
                else:
                    self.patience_counter += 1
                    print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
            
            # Save checkpoint periodically
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Best mAP: {self.best_map:.4f}")
        print(f"Results saved to: {self.save_dir}")
        print("="*60 + "\n")
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'history': self.history,
            'config': self.config
        }
        
        if is_best:
            save_path = self.save_dir / 'best_model.pth'
        else:
            save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, save_path)
        
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_map = checkpoint['best_map']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Best mAP: {self.best_map:.4f}")
