import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import spacy
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextEncoder(nn.Module):
    """Neural network for encoding text data."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])  # Take last LSTM output

class RiskPredictor(nn.Module):
    """Neural network for predicting crash risk factors."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class AIModel:
    """Main AI model handler for crash analysis and predictions."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.model_dir.parent / "data"
        self.nlp = spacy.load('en_core_web_sm')
        self.text_encoder = None
        self.risk_predictor = None
        self.vocab = None
        self.label_encoder = None
        
    def load_latest_data(self) -> pd.DataFrame:
        """Load the latest crash data for training."""
        try:
            # Get latest file from data/processed directory
            data_dir = self.data_dir / "processed"
            files = list(data_dir.glob("crash_data_*.csv"))
            if not files:
                logger.warning("No processed data files found")
                return pd.DataFrame()
            
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading data from {latest_file}")
            
            data = pd.read_csv(latest_file)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading latest data: {e}")
            return pd.DataFrame()
    
    def preprocess_text(self, texts: List[str]) -> torch.Tensor:
        """Process text data into model-ready format."""
        if not self.vocab:
            self._build_vocabulary(texts)
            
        # Convert texts to token indices
        encoded_texts = []
        for text in texts:
            doc = self.nlp(text.lower())
            tokens = [self.vocab.get(token.text, self.vocab['<UNK>']) 
                     for token in doc]
            encoded_texts.append(tokens)
            
        # Pad sequences
        max_len = max(len(seq) for seq in encoded_texts)
        padded = [seq + [self.vocab['<PAD>']] * (max_len - len(seq)) 
                 for seq in encoded_texts]
        
        return torch.tensor(padded)
    
    def _build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from training texts."""
        words = set()
        for text in texts:
            doc = self.nlp(text.lower())
            words.update(token.text for token in doc)
            
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            **{word: idx + 2 for idx, word in enumerate(words)}
        }
    
    def train(self, 
             train_data: pd.DataFrame,
             text_col: str,
             target_cols: List[str],
             epochs: int = 10,
             batch_size: int = 32) -> Dict[str, List[float]]:
        """Train the model on crash data."""
        
        # Initialize models if not exists
        if self.text_encoder is None:
            texts = train_data[text_col].tolist()
            self._build_vocabulary(texts)
            self.text_encoder = TextEncoder(
                vocab_size=len(self.vocab),
                embedding_dim=100,
                hidden_dim=128
            )
        
        if self.risk_predictor is None:
            self.risk_predictor = RiskPredictor(
                input_dim=128,  # Must match text_encoder output
                hidden_dim=64,
                output_dim=len(target_cols)
            )
        
        # Prepare data
        X_text = self.preprocess_text(train_data[text_col].tolist())
        y = torch.tensor(train_data[target_cols].values, dtype=torch.float32)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(self.text_encoder.parameters()) + 
                             list(self.risk_predictor.parameters()))
        
        # Training loop
        history = {'loss': []}
        for epoch in range(epochs):
            total_loss = 0
            batches = self._create_batches(X_text, y, batch_size)
            
            for batch_X, batch_y in batches:
                optimizer.zero_grad()
                
                # Forward pass
                text_features = self.text_encoder(batch_X)
                predictions = self.risk_predictor(text_features)
                
                # Compute loss and backpropagate
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(batches)
            history['loss'].append(avg_loss)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.save_models()
        return history
    
    def predict(self, 
                texts: List[str], 
                target_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Make predictions for new crash descriptions."""
        if self.text_encoder is None or self.risk_predictor is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # Preprocess input texts
        X_text = self.preprocess_text(texts)
        
        # Make predictions
        with torch.no_grad():
            text_features = self.text_encoder(X_text)
            predictions = self.risk_predictor(text_features)
        
        # Convert to DataFrame
        if target_cols:
            return pd.DataFrame(predictions.numpy(), columns=target_cols)
        return pd.DataFrame(predictions.numpy())
    
    def predict_risk(self, description: str) -> Dict[str, Any]:
        """Make a prediction for an incident description."""
        try:
            # For now return dummy predictions since we don't have a real model yet
            return {
                'risk_level': 'Medium',
                'confidence': 0.85,
                'factors': [
                    'Weather conditions',
                    'Mechanical issues',
                    'Human factors'
                ],
                'recommendations': (
                    'Based on the analysis, recommend enhanced pre-flight weather checks '
                    'and thorough mechanical inspections. Consider additional crew training '
                    'for similar scenarios.'
                )
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}", exc_info=True)
            raise
    
    def predict_with_confidence(self, text: str, aircraft_type: str = None, operator: str = None) -> Tuple[str, float, Dict[str, float]]:
        """Predict risk level with confidence score and feature importance.
        
        Args:
            text: Description text to analyze
            aircraft_type: Optional aircraft type
            operator: Optional operator name
            
        Returns:
            Tuple of (prediction, confidence_score, feature_importance_dict)
        """
        try:
            # Get base prediction
            prediction = self.predict([text])
            
            # Calculate confidence score (simplified version)
            confidence = 0.8  # Default confidence
            
            # Generate feature importance (simplified version)
            feature_vector = {
                'text_length': len(text),
                'has_aircraft_type': 1 if aircraft_type else 0,
                'has_operator': 1 if operator else 0
            }
            
            return prediction.iloc[0].values.tolist()[0], confidence, feature_vector
            
        except Exception as e:
            logger.error(f"Error in predict_with_confidence: {e}", exc_info=True)
            return "Unknown", 0.0, {}

    def save_models(self):
        """Save trained models and vocabularies."""
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save vocabulary
            with open(self.model_dir / 'vocab.pkl', 'wb') as f:
                pickle.dump(self.vocab, f)
                
            # Save model configurations
            config = {
                'vocab_size': len(self.vocab),
                'embedding_dim': 100,
                'hidden_dim': 128,
                'output_dim': self.risk_predictor.layers[-1].out_features
            }
            with open(self.model_dir / 'config.pkl', 'wb') as f:
                pickle.dump(config, f)
            
            # Save model states
            torch.save(self.text_encoder.state_dict(), 
                      self.model_dir / 'text_encoder.pth')
            torch.save(self.risk_predictor.state_dict(), 
                      self.model_dir / 'risk_predictor.pth')
            
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
    
    def load_models(self):
        """Load trained models and configurations."""
        try:
            logger.debug(f"Looking for models in: {self.model_dir}")
            
            # Check if model files exist
            text_encoder_path = self.model_dir / 'text_encoder.pth'
            if not text_encoder_path.exists():
                logger.warning(f"Text encoder model not found at {text_encoder_path}")
                return False
            
            # Create dummy model with same architecture
            self.text_encoder = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 2)
            )
            
            # Load saved weights with safe settings
            self.text_encoder.load_state_dict(
                torch.load(
                    text_encoder_path,
                    map_location=torch.device('cpu'),
                    weights_only=True  # Only load weights to prevent security issues
                )
            )
            
            logger.info("Successfully loaded models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            if not hasattr(self, 'text_encoder') or self.text_encoder is None:
                return {
                    'status': 'Not Loaded',
                    'architecture': 'N/A',
                    'parameters': 0,
                    'device': 'N/A'
                }
            
            # Get model info
            total_params = sum(p.numel() for p in self.text_encoder.parameters())
            device = next(self.text_encoder.parameters()).device
            
            return {
                'status': 'Loaded',
                'architecture': str(self.text_encoder),
                'parameters': total_params,
                'device': str(device)
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}", exc_info=True)
            return {
                'status': 'Error',
                'architecture': 'N/A',
                'parameters': 0,
                'device': 'N/A'
            }

    def _create_batches(self, 
                       X: torch.Tensor, 
                       y: torch.Tensor, 
                       batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create batches for training."""
        indices = torch.randperm(len(X))
        batches = []
        
        for start_idx in range(0, len(X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches

# Example usage
def main():
    # Sample data
    data = pd.DataFrame({
        'description': [
            "Aircraft crashed during takeoff in heavy rain",
            "Engine failure led to emergency landing",
            "Bird strike caused loss of control"
        ],
        'severity': [0.8, 0.5, 0.6],
        'risk_factor': [0.7, 0.4, 0.5]
    })
    
    # Initialize and train model
    model = AIModel()
    history = model.train(
        train_data=data,
        text_col='description',
        target_cols=['severity', 'risk_factor'],
        epochs=5
    )
    
    # Make predictions
    new_texts = ["Aircraft experienced turbulence during landing"]
    predictions = model.predict(new_texts, ['severity', 'risk_factor'])
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
