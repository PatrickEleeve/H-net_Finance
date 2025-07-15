#!/usr/bin/env python3
"""
H-Net for Stock Market Real-time Analysis
Based on "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"
Adapted for financial time series analysis with multi-modal data fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import math
import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HNetConfig:
    """Configuration for H-Net Stock Analysis Model"""
    # Model architecture
    d_model: int = 512
    num_stages: int = 2
    encoder_layers: int = 4
    decoder_layers: int = 4
    main_layers: int = 16
    
    # Dynamic chunking parameters
    chunk_ratios: List[int] = None  # [4, 3] for 2-stage
    smoothing_factor: float = 0.5
    
    # Stock-specific parameters  
    price_features: int = 6  # OHLCVA
    technical_features: int = 20  # Technical indicators
    news_embed_dim: int = 768  # News sentiment embeddings
    sequence_length: int = 60  # Historical data points (from our generated data)
    prediction_horizon: int = 5  # Predict next 5 periods (from our generated data)
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.1
    max_epochs: int = 100
    warmup_steps: int = 1000
    
    def __post_init__(self):
        if self.chunk_ratios is None:
            self.chunk_ratios = [4, 3]

class PositionalEncoding(nn.Module):
    """Time-aware positional encoding for financial data"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MambaLayer(nn.Module):
    """Simplified Mamba-like layer for efficient sequence processing"""
    
    def __init__(self, d_model: int, expand_factor: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 3, padding=1, groups=self.d_inner)
        self.activation = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Convolution along sequence dimension
        x = x.transpose(1, 2)  # (B, L, D) -> (B, D, L)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (B, D, L) -> (B, L, D)
        
        x = self.activation(x)
        x = x * torch.sigmoid(z)
        x = self.out_proj(x)
        
        return x + residual

class DynamicChunkingLayer(nn.Module):
    """Dynamic chunking mechanism adapted for financial time series"""
    
    def __init__(self, d_model: int, chunk_ratio: int = 4):
        super().__init__()
        self.d_model = d_model
        self.chunk_ratio = chunk_ratio
        
        # Routing module - detects regime changes and important events
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.boundary_predictor = nn.Linear(d_model, 1)
        
        # Financial-specific features for boundary detection
        self.volatility_detector = nn.Linear(d_model, 1)
        self.volume_detector = nn.Linear(d_model, 1)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            compressed_x: (batch_size, compressed_len, d_model)
            boundaries: (batch_size, seq_len) - boundary probabilities
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute similarity between adjacent timesteps
        q = self.query_proj(x)  # (B, L, D)
        k = self.key_proj(x)    # (B, L, D)
        
        # Shifted similarity calculation
        q_shifted = q[:, 1:]
        k_shifted = k[:, :-1]
        
        # Cosine similarity for temporal discontinuity detection
        similarity = F.cosine_similarity(q_shifted, k_shifted, dim=-1)  # (B, L-1)
        boundary_scores = 1 - similarity  # High score = likely boundary
        
        # Add financial-specific boundary signals
        vol_signal = torch.sigmoid(self.volatility_detector(x)).squeeze(-1)  # (B, L)
        volume_signal = torch.sigmoid(self.volume_detector(x)).squeeze(-1)   # (B, L)
        
        # Combine signals for boundary detection
        combined_scores = torch.zeros(batch_size, seq_len, device=x.device)
        combined_scores[:, 0] = 1.0  # Always start with boundary
        combined_scores[:, 1:] = (boundary_scores + 
                                vol_signal[:, 1:] * 0.3 + 
                                volume_signal[:, 1:] * 0.2)
        
        # Apply sigmoid to get probabilities
        boundary_probs = torch.sigmoid(combined_scores)
        
        # Convert to discrete boundaries (differentiable through STE)
        boundaries = (boundary_probs > 0.5).float()
        boundaries = boundaries + boundary_probs - boundary_probs.detach()  # STE
        
        # Target compression ratio using auxiliary loss
        target_ratio = seq_len / self.chunk_ratio
        actual_ratio = boundaries.sum(dim=1).mean()
        ratio_loss = F.mse_loss(actual_ratio, torch.tensor(target_ratio, device=x.device))
        
        # Compress sequence based on boundaries
        compressed_x = self._compress_sequence(x, boundaries)
        
        return compressed_x, boundary_probs, ratio_loss
    
    def _compress_sequence(self, x, boundaries):
        """Compress sequence based on boundary indicators"""
        batch_size, seq_len, d_model = x.shape
        
        # Simple selection strategy: take vectors at boundary positions
        compressed_sequences = []
        
        for b in range(batch_size):
            boundary_indices = torch.nonzero(boundaries[b], as_tuple=True)[0]
            if len(boundary_indices) == 0:
                # Fallback: uniform sampling
                step = seq_len // self.chunk_ratio
                boundary_indices = torch.arange(0, seq_len, step, device=x.device)
            
            selected_vectors = x[b, boundary_indices]  # (num_boundaries, d_model)
            compressed_sequences.append(selected_vectors)
        
        # Pad to same length
        max_chunks = max(len(seq) for seq in compressed_sequences)
        compressed_x = torch.zeros(batch_size, max_chunks, d_model, device=x.device)
        
        for b, seq in enumerate(compressed_sequences):
            compressed_x[b, :len(seq)] = seq
            
        return compressed_x

class SmoothingLayer(nn.Module):
    """Smoothing layer for differentiable decompression"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, compressed_x, boundary_probs, original_length):
        """
        Args:
            compressed_x: (batch_size, num_chunks, d_model)
            boundary_probs: (batch_size, original_length)
            original_length: int
        """
        batch_size, num_chunks, d_model = compressed_x.shape
        
        # Expand compressed representations back to original length
        expanded_x = torch.zeros(batch_size, original_length, d_model, device=compressed_x.device)
        
        for b in range(batch_size):
            # Find boundary positions
            boundary_indices = torch.nonzero(boundary_probs[b] > 0.5, as_tuple=True)[0]
            
            if len(boundary_indices) == 0:
                # Uniform expansion fallback
                step = original_length // num_chunks
                for i in range(num_chunks):
                    start_idx = i * step
                    end_idx = min((i + 1) * step, original_length)
                    expanded_x[b, start_idx:end_idx] = compressed_x[b, i]
            else:
                # Assign compressed vectors to boundary regions
                chunk_idx = 0
                for i, pos in enumerate(boundary_indices):
                    if chunk_idx < num_chunks:
                        if i < len(boundary_indices) - 1:
                            next_pos = boundary_indices[i + 1]
                            expanded_x[b, pos:next_pos] = compressed_x[b, chunk_idx]
                        else:
                            expanded_x[b, pos:] = compressed_x[b, chunk_idx]
                        chunk_idx += 1
        
        # Apply exponential moving average smoothing
        smoothed_x = torch.zeros_like(expanded_x)
        smoothed_x[:, 0] = expanded_x[:, 0]
        
        for t in range(1, original_length):
            alpha = boundary_probs[:, t:t+1]  # (batch_size, 1)
            smoothed_x[:, t] = (alpha * expanded_x[:, t] + 
                              (1 - alpha) * smoothed_x[:, t-1])
        
        return smoothed_x

class HNetEncoder(nn.Module):
    """H-Net Encoder with Mamba layers"""
    
    def __init__(self, config: HNetConfig, stage: int):
        super().__init__()
        self.config = config
        self.stage = stage
        
        self.layers = nn.ModuleList([
            MambaLayer(config.d_model) for _ in range(config.encoder_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        
        return self.norm(x)

class HNetDecoder(nn.Module):
    """H-Net Decoder with Mamba layers"""
    
    def __init__(self, config: HNetConfig, stage: int):
        super().__init__()
        self.config = config
        self.stage = stage
        
        self.layers = nn.ModuleList([
            MambaLayer(config.d_model) for _ in range(config.decoder_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.residual_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, encoder_residual):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        
        x = self.norm(x)
        
        # Add encoder residual with projection
        residual = self.residual_proj(encoder_residual)
        return x + residual

class StockTransformerBlock(nn.Module):
    """Transformer block optimized for financial data"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class HNetMainNetwork(nn.Module):
    """Main network processing compressed representations"""
    
    def __init__(self, config: HNetConfig):
        super().__init__()
        self.config = config
        
        # Use Transformer blocks for main processing
        self.layers = nn.ModuleList([
            StockTransformerBlock(config.d_model, dropout=config.dropout)
            for _ in range(config.main_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class StockHNet(nn.Module):
    """H-Net adapted for stock market analysis"""
    
    def __init__(self, config: HNetConfig):
        super().__init__()
        self.config = config
        
        # Input projection layers
        self.price_proj = nn.Linear(config.price_features, config.d_model // 2)
        self.technical_proj = nn.Linear(config.technical_features, config.d_model // 4)
        self.news_proj = nn.Linear(config.news_embed_dim, config.d_model // 4)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, config.sequence_length)
        
        # Multi-stage hierarchy
        self.encoders = nn.ModuleList([
            HNetEncoder(config, stage) for stage in range(config.num_stages)
        ])
        
        self.decoders = nn.ModuleList([
            HNetDecoder(config, stage) for stage in range(config.num_stages)
        ])
        
        self.chunking_layers = nn.ModuleList([
            DynamicChunkingLayer(config.d_model, config.chunk_ratios[stage])
            for stage in range(config.num_stages)
        ])
        
        self.smoothing_layers = nn.ModuleList([
            SmoothingLayer(config.d_model) for _ in range(config.num_stages)
        ])
        
        # Main processing network
        self.main_network = HNetMainNetwork(config)
        
        # Output heads for different prediction tasks
        self.price_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.prediction_horizon)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 4, config.prediction_horizon)
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 4, config.prediction_horizon * 3)  # up/down/flat
        )
        
    def forward(self, price_data, technical_data, news_data):
        """
        Args:
            price_data: (batch_size, seq_len, price_features)
            technical_data: (batch_size, seq_len, technical_features)
            news_data: (batch_size, seq_len, news_embed_dim)
        """
        batch_size, seq_len = price_data.shape[:2]
        
        # Multi-modal input fusion
        price_emb = self.price_proj(price_data)
        tech_emb = self.technical_proj(technical_data)
        news_emb = self.news_proj(news_data)
        
        # Concatenate embeddings
        x = torch.cat([price_emb, tech_emb, news_emb], dim=-1)  # (B, L, d_model)
        x = self.pos_encoding(x)
        
        # Store encoder outputs for residual connections
        encoder_outputs = []
        boundary_losses = []
        
        # Forward pass through hierarchical stages
        current_length = seq_len
        
        for stage in range(self.config.num_stages):
            # Encoder
            x_encoded = self.encoders[stage](x)
            encoder_outputs.append(x_encoded)
            
            # Dynamic chunking
            x, boundary_probs, ratio_loss = self.chunking_layers[stage](x_encoded)
            boundary_losses.append(ratio_loss)
            current_length = x.shape[1]
        
        # Main network processing
        x = self.main_network(x)
        
        # Reverse pass through hierarchical stages
        for stage in reversed(range(self.config.num_stages)):
            # Smoothing and decompression
            target_length = encoder_outputs[stage].shape[1]
            x = self.smoothing_layers[stage](x, 
                                           torch.ones(batch_size, target_length, device=x.device) * 0.5,
                                           target_length)
            
            # Decoder with residual connection
            x = self.decoders[stage](x, encoder_outputs[stage])
        
        # Multi-task predictions
        price_pred = self.price_head(x[:, -1])  # Use last timestep
        vol_pred = self.volatility_head(x[:, -1])
        direction_pred = self.direction_head(x[:, -1]).view(batch_size, self.config.prediction_horizon, 3)
        
        predictions = {
            'price': price_pred,
            'volatility': vol_pred,
            'direction': direction_pred
        }
        
        # Calculate total boundary loss
        total_boundary_loss = sum(boundary_losses)
        
        return predictions, total_boundary_loss

class StockDataset(Dataset):
    """Stock market dataset with multi-modal features"""
    
    def __init__(self, data_path: str, config: HNetConfig, split: str = 'train'):
        self.config = config
        self.split = split
        self.data_path = data_path
        
        # Load and preprocess data
        self.price_data, self.technical_data, self.news_data, self.targets = self._load_real_data()
        
    def _load_real_data(self):
        """Load real stock market data from generated files"""
        import os
        
        # Construct file paths for the split
        split_dir = os.path.join(self.data_path, self.split)
        
        logger.info(f"Loading {self.split} data from {split_dir}")
        
        # Auto-detect file naming pattern (with or without time period suffix)
        # First try new naming pattern with time period suffix
        import glob
        price_files = glob.glob(os.path.join(split_dir, "merged_dataset_*_price.npy"))
        if price_files:
            # Extract the pattern (e.g., "1d_1y" from "merged_dataset_1d_1y_price.npy")
            base_pattern = os.path.basename(price_files[0]).replace("_price.npy", "").replace("merged_dataset_", "")
            suffix = f"_{base_pattern}" if base_pattern else ""
        else:
            suffix = ""
        
        # Load numpy arrays
        price_file = os.path.join(split_dir, f"merged_dataset{suffix}_price.npy")
        technical_file = os.path.join(split_dir, f"merged_dataset{suffix}_technical.npy")
        news_file = os.path.join(split_dir, f"merged_dataset{suffix}_news.npy")
        targets_file = os.path.join(split_dir, f"merged_dataset{suffix}_targets.npz")
        
        # Check if files exist
        for file_path in [price_file, technical_file, news_file, targets_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        price_data = np.load(price_file)
        technical_data = np.load(technical_file)
        news_data = np.load(news_file)
        targets_data = np.load(targets_file)
        
        # Extract targets
        targets = {
            'price': targets_data['price'],
            'volatility': targets_data['volatility'],
            'direction': targets_data['direction']
        }
        
        logger.info(f"Loaded data shapes:")
        logger.info(f"  Price: {price_data.shape}")
        logger.info(f"  Technical: {technical_data.shape}")
        logger.info(f"  News: {news_data.shape}")
        logger.info(f"  Targets - Price: {targets['price'].shape}")
        logger.info(f"  Targets - Volatility: {targets['volatility'].shape}")
        logger.info(f"  Targets - Direction: {targets['direction'].shape}")
        
        # Convert to tensors
        price_tensor = torch.FloatTensor(price_data)
        technical_tensor = torch.FloatTensor(technical_data)
        news_tensor = torch.FloatTensor(news_data)
        
        target_tensors = {
            'price': torch.FloatTensor(targets['price']),
            'volatility': torch.FloatTensor(targets['volatility']),
            'direction': torch.LongTensor(targets['direction'])
        }
        
        return price_tensor, technical_tensor, news_tensor, target_tensors
    
    def __len__(self):
        return len(self.price_data)
    
    def __getitem__(self, idx):
        return {
            'price': self.price_data[idx],
            'technical': self.technical_data[idx],
            'news': self.news_data[idx],
            'targets': {k: v[idx] for k, v in self.targets.items()}
        }

class StockTrainer:
    """Training pipeline for Stock H-Net"""
    
    def __init__(self, config: HNetConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = StockHNet(config).to(device)
        
        # Calculate model size
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model size: {total_params / 1e6:.1f}M parameters")
        
        if total_params > 8e9:  # 8B limit
            logger.warning(f"Model exceeds 8B parameters: {total_params / 1e9:.1f}B")
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.max_epochs,
            steps_per_epoch=1000,  # Estimate
            pct_start=0.1
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def compute_loss(self, predictions, targets, boundary_loss):
        """Compute multi-task loss"""
        # Price prediction loss (MSE)
        price_loss = self.mse_loss(predictions['price'], targets['price'])
        
        # Volatility prediction loss (MSE)
        vol_loss = self.mse_loss(predictions['volatility'], targets['volatility'])
        
        # Direction prediction loss (CrossEntropy)
        batch_size, horizon, _ = predictions['direction'].shape
        direction_pred = predictions['direction'].view(-1, 3)
        direction_target = targets['direction'].view(-1)
        direction_loss = self.ce_loss(direction_pred, direction_target)
        
        # Combine losses
        total_loss = (price_loss + 
                     vol_loss * 0.5 + 
                     direction_loss * 0.3 + 
                     boundary_loss * 0.01)
        
        return {
            'total': total_loss,
            'price': price_loss,
            'volatility': vol_loss,
            'direction': direction_loss,
            'boundary': boundary_loss
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_losses = {'total': 0, 'price': 0, 'volatility': 0, 'direction': 0, 'boundary': 0}
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            price_data = batch['price'].to(self.device)
            technical_data = batch['technical'].to(self.device)
            news_data = batch['news'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # Forward pass
            predictions, boundary_loss = self.model(price_data, technical_data, news_data)
            
            # Compute loss
            losses = self.compute_loss(predictions, targets, boundary_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}, Total Loss: {losses['total'].item():.4f}")
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        return total_losses
    
    def validate(self, dataloader):
        """Validation step"""
        self.model.eval()
        total_losses = {'total': 0, 'price': 0, 'volatility': 0, 'direction': 0, 'boundary': 0}
        
        with torch.no_grad():
            for batch in dataloader:
                price_data = batch['price'].to(self.device)
                technical_data = batch['technical'].to(self.device)
                news_data = batch['news'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                predictions, boundary_loss = self.model(price_data, technical_data, news_data)
                losses = self.compute_loss(predictions, targets, boundary_loss)
                
                for key in total_losses:
                    total_losses[key] += losses[key].item()
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        return total_losses
    
    def train(self, train_dataset, val_dataset):
        """Main training loop"""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Disable multiprocessing to avoid issues
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0  # Disable multiprocessing to avoid issues
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.max_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            
            # Training
            train_losses = self.train_epoch(train_loader)
            logger.info(f"Train - Total: {train_losses['total']:.4f}, "
                       f"Price: {train_losses['price']:.4f}, "
                       f"Vol: {train_losses['volatility']:.4f}, "
                       f"Dir: {train_losses['direction']:.4f}")
            
            # Validation
            val_losses = self.validate(val_loader)
            logger.info(f"Val - Total: {val_losses['total']:.4f}, "
                       f"Price: {val_losses['price']:.4f}, "
                       f"Vol: {val_losses['volatility']:.4f}, "
                       f"Dir: {val_losses['direction']:.4f}")
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'config': self.config
                }, 'best_stock_hnet.pth')
                logger.info(f"New best model saved with val loss: {best_val_loss:.4f}")

def main():
    """Main training function"""
    # Configure model for our real data
    config = HNetConfig(
        d_model=256,  # Reduced for faster training
        num_stages=2,
        encoder_layers=3,
        decoder_layers=3,
        main_layers=8,  # Reduced for faster training
        chunk_ratios=[4, 3],
        batch_size=8,  # Reduced batch size
        learning_rate=1e-4,
        max_epochs=20,  # Start with fewer epochs
        sequence_length=60,  # Match our generated data
        prediction_horizon=5  # Match our generated data
    )
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    trainer = StockTrainer(config, device)
    
    # Create datasets using our generated data
    data_path = "stock_data_eodhd_extended"  # Path to our generated data
    
    try:
        train_dataset = StockDataset(data_path, config, 'train')
        val_dataset = StockDataset(data_path, config, 'val')
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        logger.info("Starting training...")
        trainer.train(train_dataset, val_dataset)
        
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.error("Please run the data preprocessing script first!")
        return
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

class RealTimeInference:
    """Real-time inference pipeline for stock market analysis"""
    
    def __init__(self, model_path: str, config: HNetConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=device)
        self.model = StockHNet(config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Data buffer for streaming inference
        self.price_buffer = np.zeros((config.sequence_length, config.price_features))
        self.technical_buffer = np.zeros((config.sequence_length, config.technical_features))
        self.news_buffer = np.zeros((config.sequence_length, config.news_embed_dim))
        
        logger.info("Real-time inference pipeline initialized")
    
    def update_data(self, new_price, new_technical, new_news):
        """Update data buffers with new incoming data"""
        # Shift buffers and add new data
        self.price_buffer[:-1] = self.price_buffer[1:]
        self.technical_buffer[:-1] = self.technical_buffer[1:]
        self.news_buffer[:-1] = self.news_buffer[1:]
        
        self.price_buffer[-1] = new_price
        self.technical_buffer[-1] = new_technical
        self.news_buffer[-1] = new_news
    
    def predict(self):
        """Make real-time predictions"""
        with torch.no_grad():
            # Convert to tensors
            price_tensor = torch.FloatTensor(self.price_buffer).unsqueeze(0).to(self.device)
            technical_tensor = torch.FloatTensor(self.technical_buffer).unsqueeze(0).to(self.device)
            news_tensor = torch.FloatTensor(self.news_buffer).unsqueeze(0).to(self.device)
            
            # Get predictions
            predictions, _ = self.model(price_tensor, technical_tensor, news_tensor)
            
            # Convert to numpy for easier handling
            results = {
                'price_forecast': predictions['price'].cpu().numpy()[0],
                'volatility_forecast': predictions['volatility'].cpu().numpy()[0],
                'direction_probs': torch.softmax(predictions['direction'], dim=-1).cpu().numpy()[0]
            }
            
            return results

class ModelOptimizer:
    """Model optimization utilities for deployment"""
    
    @staticmethod
    def quantize_model(model, dataloader, device='cuda'):
        """Apply dynamic quantization for faster inference"""
        model.eval()
        
        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with sample data
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 100:  # Use 100 batches for calibration
                    break
                
                price_data = batch['price'].to(device)
                technical_data = batch['technical'].to(device)
                news_data = batch['news'].to(device)
                
                model(price_data, technical_data, news_data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        return quantized_model
    
    @staticmethod
    def export_to_onnx(model, config, output_path='stock_hnet.onnx'):
        """Export model to ONNX format for deployment"""
        model.eval()
        
        # Create dummy inputs
        dummy_price = torch.randn(1, config.sequence_length, config.price_features)
        dummy_technical = torch.randn(1, config.sequence_length, config.technical_features)
        dummy_news = torch.randn(1, config.sequence_length, config.news_embed_dim)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_price, dummy_technical, dummy_news),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['price_data', 'technical_data', 'news_data'],
            output_names=['predictions'],
            dynamic_axes={
                'price_data': {0: 'batch_size'},
                'technical_data': {0: 'batch_size'},
                'news_data': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Model exported to {output_path}")

class AdvancedDataProcessor:
    """Advanced data processing for financial features"""
    
    @staticmethod
    def compute_technical_indicators(price_data):
        """Compute technical indicators from OHLCV data"""
        indicators = np.zeros((len(price_data), 20))  # 20 technical features
        
        # Price data: [Open, High, Low, Close, Volume, Adj_Close]
        high = price_data[:, 1]
        low = price_data[:, 2]
        close = price_data[:, 3]
        volume = price_data[:, 4]
        
        # Simple Moving Averages
        for i, window in enumerate([5, 10, 20, 50]):
            if len(close) >= window:
                sma = np.convolve(close, np.ones(window)/window, mode='same')
                indicators[:, i] = sma
        
        # Exponential Moving Averages
        for i, span in enumerate([12, 26]):
            ema = close.copy()
            alpha = 2 / (span + 1)
            for j in range(1, len(ema)):
                ema[j] = alpha * close[j] + (1 - alpha) * ema[j-1]
            indicators[:, 4 + i] = ema
        
        # RSI (Relative Strength Index)
        rsi_period = 14
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(rsi_period)/rsi_period, mode='same')
        avg_loss = np.convolve(loss, np.ones(rsi_period)/rsi_period, mode='same')
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        indicators[1:, 6] = rsi
        
        # MACD
        ema12 = indicators[:, 4]
        ema26 = indicators[:, 5]
        macd_line = ema12 - ema26
        signal_line = np.convolve(macd_line, np.ones(9)/9, mode='same')
        histogram = macd_line - signal_line
        
        indicators[:, 7] = macd_line
        indicators[:, 8] = signal_line
        indicators[:, 9] = histogram
        
        # Bollinger Bands
        bb_period = 20
        bb_std_dev = 2
        sma20 = indicators[:, 2]  # 20-period SMA
        rolling_std = np.zeros_like(close)
        
        for i in range(bb_period, len(close)):
            rolling_std[i] = np.std(close[i-bb_period:i])
        
        bb_upper = sma20 + (rolling_std * bb_std_dev)
        bb_lower = sma20 - (rolling_std * bb_std_dev)
        bb_percent = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        indicators[:, 10] = bb_upper
        indicators[:, 11] = bb_lower
        indicators[:, 12] = bb_percent
        
        # Volume indicators
        # Volume SMA
        vol_sma = np.convolve(volume, np.ones(20)/20, mode='same')
        indicators[:, 13] = vol_sma
        
        # Volume ratio
        indicators[:, 14] = volume / (vol_sma + 1e-10)
        
        # Price volatility (rolling standard deviation)
        for i in range(20, len(close)):
            indicators[i, 15] = np.std(close[i-20:i])
        
        # Price momentum
        for i, period in enumerate([1, 5, 10]):
            momentum = np.zeros_like(close)
            momentum[period:] = (close[period:] - close[:-period]) / close[:-period]
            indicators[:, 16 + i] = momentum
        
        # High-Low spread
        indicators[:, 19] = (high - low) / close
        
        return indicators
    
    @staticmethod
    def process_news_sentiment(news_texts, model_name='finbert'):
        """Process news sentiment using FinBERT or similar model"""
        # Placeholder for news sentiment processing
        # In practice, you would use a pre-trained financial sentiment model
        
        # Simulate sentiment embeddings
        batch_size = len(news_texts)
        sentiment_embeddings = np.random.randn(batch_size, 768)  # BERT-like embeddings
        
        return sentiment_embeddings
    
    @staticmethod
    def create_market_regime_features(price_data, lookback=252):
        """Create market regime classification features"""
        close_prices = price_data[:, 3]  # Close prices
        returns = np.diff(np.log(close_prices))
        
        # Volatility regime
        rolling_vol = np.zeros_like(close_prices)
        for i in range(lookback, len(close_prices)):
            rolling_vol[i] = np.std(returns[i-lookback:i]) * np.sqrt(252)
        
        # Trend regime (using moving average slopes)
        ma_50 = np.convolve(close_prices, np.ones(50)/50, mode='same')
        ma_200 = np.convolve(close_prices, np.ones(200)/200, mode='same')
        
        trend_signal = np.where(ma_50 > ma_200, 1, -1)  # 1 = uptrend, -1 = downtrend
        
        # Market stress indicator (VIX-like)
        stress_indicator = rolling_vol / np.mean(rolling_vol[-252:])
        
        regime_features = np.column_stack([
            rolling_vol,
            trend_signal,
            stress_indicator
        ])
        
        return regime_features

class EvaluationMetrics:
    """Evaluation metrics for financial forecasting"""
    
    @staticmethod
    def directional_accuracy(predictions, actuals):
        """Calculate directional accuracy"""
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        
        accuracy = np.mean(pred_direction == actual_direction)
        return accuracy
    
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)
    
    @staticmethod
    def max_drawdown(cumulative_returns):
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)
    
    @staticmethod
    def information_ratio(portfolio_returns, benchmark_returns):
        """Calculate information ratio"""
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(active_returns)
        
        if tracking_error == 0:
            return 0
        
        return np.mean(active_returns) / tracking_error
    
    @staticmethod
    def evaluate_model_performance(model, test_dataloader, device='cuda'):
        """Comprehensive model evaluation"""
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                price_data = batch['price'].to(device)
                technical_data = batch['technical'].to(device)
                news_data = batch['news'].to(device)
                targets = batch['targets']
                
                predictions, _ = model(price_data, technical_data, news_data)
                
                all_predictions.append({
                    'price': predictions['price'].cpu().numpy(),
                    'volatility': predictions['volatility'].cpu().numpy(),
                    'direction': torch.softmax(predictions['direction'], dim=-1).cpu().numpy()
                })
                
                all_targets.append({
                    'price': targets['price'].numpy(),
                    'volatility': targets['volatility'].numpy(),
                    'direction': targets['direction'].numpy()
                })
        
        # Aggregate results
        pred_prices = np.concatenate([p['price'] for p in all_predictions])
        true_prices = np.concatenate([t['price'] for t in all_targets])
        
        pred_vols = np.concatenate([p['volatility'] for p in all_predictions])
        true_vols = np.concatenate([t['volatility'] for t in all_targets])
        
        pred_directions = np.concatenate([p['direction'] for p in all_predictions])
        true_directions = np.concatenate([t['direction'] for t in all_targets])
        
        # Calculate metrics
        metrics = {
            'price_mse': np.mean((pred_prices - true_prices) ** 2),
            'price_mae': np.mean(np.abs(pred_prices - true_prices)),
            'vol_mse': np.mean((pred_vols - true_vols) ** 2),
            'directional_accuracy': EvaluationMetrics.directional_accuracy(
                pred_prices.mean(axis=1), true_prices.mean(axis=1)
            ),
            'direction_accuracy': np.mean(
                np.argmax(pred_directions, axis=-1) == true_directions
            )
        }
        
        return metrics

def hyperparameter_search():
    """Automated hyperparameter optimization"""
    import optuna
    
    def objective(trial):
        # Suggest hyperparameters
        config = HNetConfig(
            d_model=trial.suggest_categorical('d_model', [256, 512, 768]),
            encoder_layers=trial.suggest_int('encoder_layers', 2, 6),
            decoder_layers=trial.suggest_int('decoder_layers', 2, 6),
            main_layers=trial.suggest_int('main_layers', 8, 24),
            learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            dropout=trial.suggest_uniform('dropout', 0.1, 0.3),
            chunk_ratios=[
                trial.suggest_int('chunk_ratio_0', 2, 6),
                trial.suggest_int('chunk_ratio_1', 2, 4)
            ]
        )
        
        # Quick training with subset of data
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = StockTrainer(config, device)
        
        # Create smaller datasets for hyperparameter search
        train_dataset = StockDataset('data/train/', config, 'train')
        val_dataset = StockDataset('data/val/', config, 'val')
        
        # Train for fewer epochs
        config.max_epochs = 5
        trainer.train(train_dataset, val_dataset)
        
        # Return validation loss
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        val_losses = trainer.validate(val_loader)
        
        return val_losses['total']
    
    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    logger.info(f"Best parameters: {study.best_params}")
    return study.best_params

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='H-Net Stock Market Analysis')
    parser.add_argument('--mode', choices=['train', 'inference', 'optimize'], 
                       default='train', help='Mode to run')
    parser.add_argument('--model_path', type=str, default='best_stock_hnet.pth',
                       help='Path to saved model')
    parser.add_argument('--data_path', type=str, default='data/',
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        main()
    elif args.mode == 'inference':
        config = HNetConfig()
        inference = RealTimeInference(args.model_path, config)
        logger.info("Real-time inference ready")
    elif args.mode == 'optimize':
        best_params = hyperparameter_search()
        logger.info(f"Optimization complete. Best parameters: {best_params}")
    
    logger.info("Done!")
    