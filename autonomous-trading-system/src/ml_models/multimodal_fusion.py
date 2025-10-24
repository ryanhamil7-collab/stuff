"""
Multimodal Fusion: BERT + LSTM

Combines:
1. BERT for news sentiment analysis
2. LSTM for price pattern recognition
3. Fusion layer for joint predictions

Target: Improved predictions by combining text and price data
Optimized for GPU (A100/L4)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from src.utils import log, config


class BERTSentimentAnalyzer(nn.Module):
    """
    BERT-based Sentiment Analyzer
    
    Uses pre-trained BERT (or FinBERT) for financial news sentiment.
    
    Features:
    - Fine-tuning on financial data
    - Multi-class sentiment (positive, negative, neutral)
    - Attention weights for interpretability
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize BERT sentiment analyzer
        
        Args:
            model_name: Pre-trained model name (e.g., 'bert-base-uncased', 'ProsusAI/finbert')
            num_classes: Number of sentiment classes
            dropout: Dropout rate
        """
        super(BERTSentimentAnalyzer, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Sentiment logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def encode_text(self, texts: List[str], max_length: int = 128) -> Dict:
        """
        Tokenize and encode texts
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Encoded inputs
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encoded


class MultimodalFusionModel(nn.Module):
    """
    Multimodal Fusion: BERT + LSTM
    
    Architecture:
    1. BERT branch: Processes news text
    2. LSTM branch: Processes price sequences
    3. Fusion layer: Combines both modalities
    4. Prediction head: Final output
    
    Features:
    - Early fusion (concatenation)
    - Late fusion (separate predictions + ensemble)
    - Attention-based fusion
    """
    
    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        lstm_input_size: int = 10,
        lstm_hidden_size: int = 128,
        fusion_method: str = 'concat',
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        """
        Initialize multimodal fusion model
        
        Args:
            bert_model_name: Pre-trained BERT model
            lstm_input_size: LSTM input features
            lstm_hidden_size: LSTM hidden size
            fusion_method: 'concat', 'attention', or 'late'
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(MultimodalFusionModel, self).__init__()
        
        self.fusion_method = fusion_method
        
        self.bert_sentiment = BERTSentimentAnalyzer(
            model_name=bert_model_name,
            num_classes=num_classes,
            dropout=dropout
        )
        bert_output_size = self.bert_sentiment.bert.config.hidden_size
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        self.lstm_dropout = nn.Dropout(dropout)
        
        if fusion_method == 'concat':
            fusion_input_size = bert_output_size + lstm_hidden_size
            self.fusion_fc = nn.Linear(fusion_input_size, lstm_hidden_size)
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.bert_projection = nn.Linear(bert_output_size, lstm_hidden_size)
        elif fusion_method == 'late':
            self.bert_head = nn.Linear(bert_output_size, num_classes)
            self.lstm_head = nn.Linear(lstm_hidden_size, num_classes)
        
        if fusion_method != 'late':
            self.output_fc = nn.Linear(lstm_hidden_size, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(
        self,
        input_ids,
        attention_mask,
        price_sequence
    ):
        """
        Forward pass
        
        Args:
            input_ids: BERT token IDs (batch_size, seq_len)
            attention_mask: BERT attention mask
            price_sequence: LSTM price sequence (batch_size, seq_len, features)
            
        Returns:
            Predictions
        """
        bert_outputs = self.bert_sentiment.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        bert_features = bert_outputs.pooler_output  # (batch_size, bert_hidden)
        
        lstm_out, (h_n, c_n) = self.lstm(price_sequence)
        lstm_features = lstm_out[:, -1, :]  # (batch_size, lstm_hidden)
        lstm_features = self.lstm_dropout(lstm_features)
        
        if self.fusion_method == 'concat':
            fused = torch.cat([bert_features, lstm_features], dim=1)
            fused = self.fusion_fc(fused)
            fused = self.relu(fused)
            fused = self.dropout(fused)
            
            output = self.output_fc(fused)
        
        elif self.fusion_method == 'attention':
            bert_proj = self.bert_projection(bert_features).unsqueeze(1)  # (batch_size, 1, lstm_hidden)
            lstm_proj = lstm_features.unsqueeze(1)  # (batch_size, 1, lstm_hidden)
            
            fused, _ = self.attention(
                query=lstm_proj,
                key=bert_proj,
                value=bert_proj
            )
            fused = fused.squeeze(1)  # (batch_size, lstm_hidden)
            fused = self.dropout(fused)
            
            output = self.output_fc(fused)
        
        elif self.fusion_method == 'late':
            bert_pred = self.bert_head(bert_features)
            lstm_pred = self.lstm_head(lstm_features)
            
            output = (bert_pred + lstm_pred) / 2
        
        return output


class MultimodalTrader:
    """
    Multimodal Trading System
    
    Combines news sentiment and price patterns for trading decisions.
    
    Features:
    - News fetching and preprocessing
    - Sentiment analysis with BERT
    - Price pattern recognition with LSTM
    - Joint prediction with fusion
    - GPU acceleration
    """
    
    def __init__(
        self,
        bert_model: str = 'ProsusAI/finbert',
        fusion_method: str = 'concat',
        use_gpu: bool = True
    ):
        """
        Initialize multimodal trader
        
        Args:
            bert_model: Pre-trained BERT model
            fusion_method: 'concat', 'attention', or 'late'
            use_gpu: Use GPU if available
        """
        self.config = config.get('ml.multimodal', {
            'bert_model': 'ProsusAI/finbert',
            'lstm_input_size': 10,
            'lstm_hidden_size': 128,
            'fusion_method': 'concat',
            'num_classes': 3,
            'dropout': 0.2,
            'learning_rate': 0.00001,
            'batch_size': 16,
            'num_epochs': 10,
            'max_text_length': 128,
            'use_gpu': True
        })
        
        self.bert_model = bert_model or self.config['bert_model']
        self.fusion_method = fusion_method or self.config['fusion_method']
        self.use_gpu = use_gpu and self.config['use_gpu']
        self.device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.tokenizer = None
        
        log.info(f"MultimodalTrader initialized - BERT: {self.bert_model}, Fusion: {self.fusion_method}, Device: {self.device}")
    
    def build_model(self, lstm_input_size: int):
        """Build multimodal fusion model"""
        self.model = MultimodalFusionModel(
            bert_model_name=self.bert_model,
            lstm_input_size=lstm_input_size,
            lstm_hidden_size=self.config['lstm_hidden_size'],
            fusion_method=self.fusion_method,
            num_classes=self.config['num_classes'],
            dropout=self.config['dropout']
        )
        
        self.model = self.model.to(self.device)
        
        self.tokenizer = self.model.bert_sentiment.tokenizer
        
        num_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"Built multimodal model with {num_params:,} parameters")
    
    def prepare_data(
        self,
        texts: List[str],
        price_sequences: np.ndarray,
        labels: np.ndarray
    ) -> Tuple:
        """
        Prepare multimodal data
        
        Args:
            texts: News texts
            price_sequences: Price sequences (num_samples, seq_len, features)
            labels: Labels
            
        Returns:
            Prepared tensors
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config['max_text_length'],
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        price_tensor = torch.FloatTensor(price_sequences).to(self.device)
        
        labels_tensor = torch.LongTensor(labels).to(self.device)
        
        return input_ids, attention_mask, price_tensor, labels_tensor
    
    def train(
        self,
        train_texts: List[str],
        train_prices: np.ndarray,
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_prices: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train multimodal model
        
        Args:
            train_texts: Training news texts
            train_prices: Training price sequences
            train_labels: Training labels
            val_texts: Validation texts
            val_prices: Validation prices
            val_labels: Validation labels
            
        Returns:
            Training statistics
        """
        if self.model is None:
            self.build_model(train_prices.shape[2])
        
        log.info("Starting multimodal training...")
        
        train_input_ids, train_attention_mask, train_price_tensor, train_labels_tensor = self.prepare_data(
            train_texts, train_prices, train_labels
        )
        
        if val_texts is not None:
            val_input_ids, val_attention_mask, val_price_tensor, val_labels_tensor = self.prepare_data(
                val_texts, val_prices, val_labels
            )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
        batch_size = self.config['batch_size']
        num_epochs = self.config['num_epochs']
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(train_input_ids), batch_size):
                batch_input_ids = train_input_ids[i:i+batch_size]
                batch_attention_mask = train_attention_mask[i:i+batch_size]
                batch_prices = train_price_tensor[i:i+batch_size]
                batch_labels = train_labels_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    price_sequence=batch_prices
                )
                
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            if val_texts is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(
                        input_ids=val_input_ids,
                        attention_mask=val_attention_mask,
                        price_sequence=val_price_tensor
                    )
                    val_loss = criterion(val_outputs, val_labels_tensor).item()
                    val_losses.append(val_loss)
                
                log.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                log.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        log.info("Multimodal training complete")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses if val_texts is not None else None
        }
    
    def predict(
        self,
        texts: List[str],
        price_sequences: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions
        
        Args:
            texts: News texts
            price_sequences: Price sequences
            
        Returns:
            Predictions
        """
        self.model.eval()
        
        input_ids, attention_mask, price_tensor, _ = self.prepare_data(
            texts, price_sequences, np.zeros(len(texts))
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                price_sequence=price_tensor
            )
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        log.info(f"Model saved to {path}")
    
    def load_model(self, path: str, lstm_input_size: int):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.build_model(lstm_input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        log.info(f"Model loaded from {path}")
