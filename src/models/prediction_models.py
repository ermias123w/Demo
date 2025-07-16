import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class CryptoDataset(Dataset):
    """PyTorch Dataset for cryptocurrency data"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, 
                 sentiment_data: Optional[np.ndarray] = None):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.sentiment_data = torch.FloatTensor(sentiment_data) if sentiment_data is not None else None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.sentiment_data is not None:
            return self.sequences[idx], self.sentiment_data[idx], self.targets[idx]
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 3):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output
        output = self.dropout(lstm_out[:, -1, :])
        output = self.fc(output)
        output = self.softmax(output)
        
        return output

class TransformerModel(nn.Module):
    """Transformer model for sequence prediction"""
    
    def __init__(self, input_size: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1, output_size: int = 3):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer expects (seq_len, batch_size, d_model) but we use batch_first=True
        output = self.transformer(x)
        
        # Use the last output
        output = self.dropout(output[:, -1, :])
        output = self.fc(output)
        output = self.softmax(output)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class SentimentModel(nn.Module):
    """Simple model for sentiment-based prediction"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, output_size: int = 3):
        super(SentimentModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

class EnsembleModel(nn.Module):
    """Ensemble model combining LSTM, Transformer, and Sentiment models"""
    
    def __init__(self, lstm_model: LSTMModel, transformer_model: TransformerModel, 
                 sentiment_model: SentimentModel, ensemble_method: str = 'weighted'):
        super(EnsembleModel, self).__init__()
        
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model
        self.sentiment_model = sentiment_model
        self.ensemble_method = ensemble_method
        
        # Learned weights for ensemble
        if ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))  # LSTM, Transformer, Sentiment
        
        # Final combination layer
        self.final_layer = nn.Linear(3, 3)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, price_sequence, sentiment_data):
        # Get predictions from individual models
        lstm_output = self.lstm_model(price_sequence)
        transformer_output = self.transformer_model(price_sequence)
        sentiment_output = self.sentiment_model(sentiment_data)
        
        if self.ensemble_method == 'weighted':
            # Weighted average
            weights = torch.softmax(self.weights, dim=0)
            ensemble_output = (weights[0] * lstm_output + 
                              weights[1] * transformer_output + 
                              weights[2] * sentiment_output)
        elif self.ensemble_method == 'voting':
            # Majority voting
            ensemble_output = (lstm_output + transformer_output + sentiment_output) / 3
        else:
            # Concatenate and learn combination
            combined = torch.cat([lstm_output, transformer_output, sentiment_output], dim=1)
            ensemble_output = self.final_layer(combined)
            ensemble_output = self.softmax(ensemble_output)
        
        return ensemble_output, lstm_output, transformer_output, sentiment_output

class CryptoPredictionSystem:
    """Main prediction system coordinating all models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.lstm_model = None
        self.transformer_model = None
        self.sentiment_model = None
        self.ensemble_model = None
        
        # Data preprocessing
        self.price_scaler = StandardScaler()
        self.sentiment_scaler = StandardScaler()
        
        # Training history
        self.training_history = []
        
        # Model paths
        self.model_save_path = Path(config.get('model_save_path', 'models_checkpoints'))
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CryptoPredictionSystem initialized with device: {self.device}")
    
    def prepare_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""
        try:
            # Ensure we have required columns
            required_columns = ['price', 'volume', 'sentiment_score']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
                return None, None, None
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Prepare features
            feature_columns = [
                'price', 'volume', 'market_cap', 'price_change_24h',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'ema_10', 'ema_50', 'ema_200',
                'bb_upper', 'bb_middle', 'bb_lower',
                'atr', 'volatility', 'volume_spike'
            ]
            
            # Filter available columns
            available_features = [col for col in feature_columns if col in df.columns]
            
            # Fill missing values
            df[available_features] = df[available_features].fillna(method='ffill').fillna(method='bfill')
            
            # Create targets (price direction)
            df['price_change'] = df['price'].pct_change()
            df['target'] = 0  # HOLD
            df.loc[df['price_change'] > 0.001, 'target'] = 2  # BUY
            df.loc[df['price_change'] < -0.001, 'target'] = 1  # SELL
            
            # Scale features
            scaled_features = self.price_scaler.fit_transform(df[available_features])
            
            # Scale sentiment
            sentiment_data = df['sentiment_score'].values.reshape(-1, 1)
            scaled_sentiment = self.sentiment_scaler.fit_transform(sentiment_data)
            
            # Create sequences
            sequences = []
            sentiment_sequences = []
            targets = []
            
            for i in range(sequence_length, len(df)):
                # Price sequence
                seq = scaled_features[i-sequence_length:i]
                sequences.append(seq)
                
                # Sentiment sequence (average over sequence)
                sent_seq = scaled_sentiment[i-sequence_length:i].mean()
                sentiment_sequences.append([sent_seq])
                
                # Target
                targets.append(df.iloc[i]['target'])
            
            sequences = np.array(sequences)
            sentiment_sequences = np.array(sentiment_sequences)
            targets = np.array(targets)
            
            logger.info(f"Prepared {len(sequences)} sequences with {len(available_features)} features")
            
            return sequences, sentiment_sequences, targets
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None, None
    
    def build_models(self, input_size: int):
        """Build all models"""
        try:
            # LSTM Model
            self.lstm_model = LSTMModel(
                input_size=input_size,
                hidden_size=self.config.get('lstm_hidden_size', 128),
                num_layers=self.config.get('lstm_num_layers', 2),
                dropout=self.config.get('lstm_dropout', 0.2)
            ).to(self.device)
            
            # Transformer Model
            self.transformer_model = TransformerModel(
                input_size=input_size,
                d_model=self.config.get('transformer_d_model', 512),
                nhead=self.config.get('transformer_nhead', 8),
                num_layers=self.config.get('transformer_num_layers', 6),
                dropout=self.config.get('transformer_dropout', 0.1)
            ).to(self.device)
            
            # Sentiment Model
            self.sentiment_model = SentimentModel(
                input_size=1,
                hidden_size=64
            ).to(self.device)
            
            # Ensemble Model
            self.ensemble_model = EnsembleModel(
                self.lstm_model,
                self.transformer_model,
                self.sentiment_model,
                ensemble_method='weighted'
            ).to(self.device)
            
            logger.info("All models built successfully")
            
        except Exception as e:
            logger.error(f"Error building models: {e}")
            raise
    
    def train_models(self, sequences: np.ndarray, sentiment_data: np.ndarray, 
                    targets: np.ndarray, epochs: int = 100) -> Dict[str, Any]:
        """Train all models"""
        try:
            # Split data
            train_size = int(0.8 * len(sequences))
            val_size = int(0.1 * len(sequences))
            
            train_seq = sequences[:train_size]
            train_sent = sentiment_data[:train_size]
            train_targets = targets[:train_size]
            
            val_seq = sequences[train_size:train_size+val_size]
            val_sent = sentiment_data[train_size:train_size+val_size]
            val_targets = targets[train_size:train_size+val_size]
            
            # Create datasets
            train_dataset = CryptoDataset(train_seq, train_targets, train_sent)
            val_dataset = CryptoDataset(val_seq, val_targets, val_sent)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.get('batch_size', 32), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.get('batch_size', 32), shuffle=False)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                self.ensemble_model.parameters(),
                lr=self.config.get('learning_rate', 0.001)
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience = self.config.get('early_stopping_patience', 10)
            patience_counter = 0
            
            train_losses = []
            val_losses = []
            val_accuracies = []
            
            for epoch in range(epochs):
                # Training
                self.ensemble_model.train()
                train_loss = 0.0
                
                for batch_idx, (price_seq, sent_data, targets_batch) in enumerate(train_loader):
                    price_seq = price_seq.to(self.device)
                    sent_data = sent_data.to(self.device)
                    targets_batch = targets_batch.long().to(self.device)
                    
                    optimizer.zero_grad()
                    
                    ensemble_output, lstm_output, transformer_output, sentiment_output = \
                        self.ensemble_model(price_seq, sent_data)
                    
                    # Loss for ensemble
                    loss = criterion(ensemble_output, targets_batch)
                    
                    # Add individual model losses for regularization
                    loss += 0.1 * criterion(lstm_output, targets_batch)
                    loss += 0.1 * criterion(transformer_output, targets_batch)
                    loss += 0.1 * criterion(sentiment_output, targets_batch)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.ensemble_model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for price_seq, sent_data, targets_batch in val_loader:
                        price_seq = price_seq.to(self.device)
                        sent_data = sent_data.to(self.device)
                        targets_batch = targets_batch.long().to(self.device)
                        
                        ensemble_output, _, _, _ = self.ensemble_model(price_seq, sent_data)
                        
                        loss = criterion(ensemble_output, targets_batch)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(ensemble_output.data, 1)
                        total += targets_batch.size(0)
                        correct += (predicted == targets_batch).sum().item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * correct / total
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                val_accuracies.append(val_accuracy)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                               f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_models(f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f'Early stopping at epoch {epoch+1}')
                        break
            
            training_results = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss,
                'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0
            }
            
            self.training_history.append(training_results)
            
            logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {}
    
    def predict(self, sequences: np.ndarray, sentiment_data: np.ndarray) -> Dict[str, Any]:
        """Make predictions using the ensemble model"""
        try:
            if self.ensemble_model is None:
                logger.error("Model not trained yet")
                return {}
            
            self.ensemble_model.eval()
            
            with torch.no_grad():
                # Convert to tensors
                price_seq = torch.FloatTensor(sequences).to(self.device)
                sent_data = torch.FloatTensor(sentiment_data).to(self.device)
                
                # Get predictions
                ensemble_output, lstm_output, transformer_output, sentiment_output = \
                    self.ensemble_model(price_seq, sent_data)
                
                # Convert to numpy
                ensemble_probs = ensemble_output.cpu().numpy()
                lstm_probs = lstm_output.cpu().numpy()
                transformer_probs = transformer_output.cpu().numpy()
                sentiment_probs = sentiment_output.cpu().numpy()
                
                # Get predicted classes
                ensemble_pred = np.argmax(ensemble_probs, axis=1)
                lstm_pred = np.argmax(lstm_probs, axis=1)
                transformer_pred = np.argmax(transformer_probs, axis=1)
                sentiment_pred = np.argmax(sentiment_probs, axis=1)
                
                # Calculate confidence (max probability)
                ensemble_confidence = np.max(ensemble_probs, axis=1)
                
                # Map predictions to labels
                label_mapping = {0: 'HOLD', 1: 'SELL', 2: 'BUY'}
                
                results = {
                    'ensemble_prediction': [label_mapping[pred] for pred in ensemble_pred],
                    'lstm_prediction': [label_mapping[pred] for pred in lstm_pred],
                    'transformer_prediction': [label_mapping[pred] for pred in transformer_pred],
                    'sentiment_prediction': [label_mapping[pred] for pred in sentiment_pred],
                    'ensemble_confidence': ensemble_confidence.tolist(),
                    'ensemble_probabilities': ensemble_probs.tolist(),
                    'lstm_probabilities': lstm_probs.tolist(),
                    'transformer_probabilities': transformer_probs.tolist(),
                    'sentiment_probabilities': sentiment_probs.tolist()
                }
                
                return results
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {}
    
    def save_models(self, model_name: str):
        """Save all models and scalers"""
        try:
            save_path = self.model_save_path / model_name
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save models
            torch.save(self.ensemble_model.state_dict(), save_path / 'ensemble_model.pth')
            torch.save(self.lstm_model.state_dict(), save_path / 'lstm_model.pth')
            torch.save(self.transformer_model.state_dict(), save_path / 'transformer_model.pth')
            torch.save(self.sentiment_model.state_dict(), save_path / 'sentiment_model.pth')
            
            # Save scalers
            joblib.dump(self.price_scaler, save_path / 'price_scaler.pkl')
            joblib.dump(self.sentiment_scaler, save_path / 'sentiment_scaler.pkl')
            
            # Save configuration
            with open(save_path / 'config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Save training history
            with open(save_path / 'training_history.json', 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            logger.info(f"Models saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, model_name: str):
        """Load saved models and scalers"""
        try:
            load_path = self.model_save_path / model_name
            
            if not load_path.exists():
                logger.error(f"Model path {load_path} does not exist")
                return False
            
            # Load configuration
            with open(load_path / 'config.json', 'r') as f:
                self.config = json.load(f)
            
            # Build models first
            # Note: You need to know the input size to build models
            # This should be saved in config or determined from data
            input_size = self.config.get('input_size', 16)  # Default value
            self.build_models(input_size)
            
            # Load model states
            self.ensemble_model.load_state_dict(torch.load(load_path / 'ensemble_model.pth', map_location=self.device))
            self.lstm_model.load_state_dict(torch.load(load_path / 'lstm_model.pth', map_location=self.device))
            self.transformer_model.load_state_dict(torch.load(load_path / 'transformer_model.pth', map_location=self.device))
            self.sentiment_model.load_state_dict(torch.load(load_path / 'sentiment_model.pth', map_location=self.device))
            
            # Load scalers
            self.price_scaler = joblib.load(load_path / 'price_scaler.pkl')
            self.sentiment_scaler = joblib.load(load_path / 'sentiment_scaler.pkl')
            
            # Load training history
            if (load_path / 'training_history.json').exists():
                with open(load_path / 'training_history.json', 'r') as f:
                    self.training_history = json.load(f)
            
            logger.info(f"Models loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def evaluate_model(self, sequences: np.ndarray, sentiment_data: np.ndarray, 
                      targets: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            predictions = self.predict(sequences, sentiment_data)
            
            if not predictions:
                return {}
            
            # Convert predictions to numerical format
            label_to_num = {'HOLD': 0, 'SELL': 1, 'BUY': 2}
            ensemble_pred_num = [label_to_num[pred] for pred in predictions['ensemble_prediction']]
            
            # Calculate metrics
            accuracy = accuracy_score(targets, ensemble_pred_num)
            precision = precision_score(targets, ensemble_pred_num, average='weighted')
            recall = recall_score(targets, ensemble_pred_num, average='weighted')
            f1 = f1_score(targets, ensemble_pred_num, average='weighted')
            
            # Calculate individual model accuracies
            lstm_pred_num = [label_to_num[pred] for pred in predictions['lstm_prediction']]
            transformer_pred_num = [label_to_num[pred] for pred in predictions['transformer_prediction']]
            sentiment_pred_num = [label_to_num[pred] for pred in predictions['sentiment_prediction']]
            
            lstm_accuracy = accuracy_score(targets, lstm_pred_num)
            transformer_accuracy = accuracy_score(targets, transformer_pred_num)
            sentiment_accuracy = accuracy_score(targets, sentiment_pred_num)
            
            evaluation_results = {
                'ensemble_accuracy': accuracy,
                'ensemble_precision': precision,
                'ensemble_recall': recall,
                'ensemble_f1': f1,
                'lstm_accuracy': lstm_accuracy,
                'transformer_accuracy': transformer_accuracy,
                'sentiment_accuracy': sentiment_accuracy,
                'average_confidence': np.mean(predictions['ensemble_confidence'])
            }
            
            logger.info(f"Model evaluation completed. Ensemble accuracy: {accuracy:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all models"""
        try:
            summary = {
                'models_built': self.ensemble_model is not None,
                'device': str(self.device),
                'training_history_length': len(self.training_history),
                'config': self.config
            }
            
            if self.ensemble_model is not None:
                # Count parameters
                lstm_params = sum(p.numel() for p in self.lstm_model.parameters() if p.requires_grad)
                transformer_params = sum(p.numel() for p in self.transformer_model.parameters() if p.requires_grad)
                sentiment_params = sum(p.numel() for p in self.sentiment_model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.ensemble_model.parameters() if p.requires_grad)
                
                summary.update({
                    'lstm_parameters': lstm_params,
                    'transformer_parameters': transformer_params,
                    'sentiment_parameters': sentiment_params,
                    'total_parameters': total_params
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {}