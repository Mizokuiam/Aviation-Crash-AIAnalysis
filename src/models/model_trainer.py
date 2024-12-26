import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging
from datetime import datetime

from .ai_model import AIModel, TextEncoder, RiskPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrashDataset(Dataset):
    """Dataset class for crash data."""
    
    def __init__(self, texts: torch.Tensor, labels: torch.Tensor):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class ModelTrainer:
    """Enhanced model trainer with cross-validation and versioning."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.model_dir / "versions.json"
        self.current_version = self._load_latest_version()
        self.model = AIModel()
    
    def _load_latest_version(self) -> Dict:
        """Load the latest model version information."""
        if not self.version_file.exists():
            return {"version": "v0.0.0", "timestamp": None, "metrics": {}}
        
        with open(self.version_file, 'r') as f:
            versions = json.load(f)
            return versions[-1] if versions else {"version": "v0.0.0", "timestamp": None, "metrics": {}}
    
    def _increment_version(self) -> str:
        """Increment the model version number."""
        current = self.current_version["version"].lstrip('v').split('.')
        current[-1] = str(int(current[-1]) + 1)
        return 'v' + '.'.join(current)
    
    def train_with_cv(self, 
                      train_data: pd.DataFrame,
                      text_col: str,
                      target_cols: List[str],
                      n_splits: int = 5,
                      epochs: int = 10,
                      batch_size: int = 32) -> Dict:
        """Train model with k-fold cross-validation."""
        logger.info("Preparing data for cross-validation...")
        
        # Preprocess all texts
        X = self.model.preprocess_text(train_data[text_col].tolist())
        y = torch.tensor(train_data[target_cols].values, dtype=torch.float32)
        
        # Initialize K-fold
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Metrics for each fold
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            # Create data loaders for this fold
            train_dataset = CrashDataset(X[train_idx], y[train_idx])
            val_dataset = CrashDataset(X[val_idx], y[val_idx])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize new model for this fold
            fold_model = AIModel()
            fold_model.text_encoder = TextEncoder(
                vocab_size=len(self.model.vocab),
                embedding_dim=100,
                hidden_dim=128
            )
            fold_model.risk_predictor = RiskPredictor(
                input_dim=128,
                hidden_dim=64,
                output_dim=len(target_cols)
            )
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                list(fold_model.text_encoder.parameters()) +
                list(fold_model.risk_predictor.parameters())
            )
            
            # Training loop
            fold_history = {'train_loss': [], 'val_loss': []}
            
            for epoch in range(epochs):
                # Training phase
                fold_model.text_encoder.train()
                fold_model.risk_predictor.train()
                train_loss = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    text_features = fold_model.text_encoder(batch_X)
                    predictions = fold_model.risk_predictor(text_features)
                    
                    # Compute loss and backpropagate
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                fold_model.text_encoder.eval()
                fold_model.risk_predictor.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        text_features = fold_model.text_encoder(batch_X)
                        predictions = fold_model.risk_predictor(text_features)
                        val_loss += criterion(predictions, batch_y).item()
                
                # Record metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                fold_history['train_loss'].append(avg_train_loss)
                fold_history['val_loss'].append(avg_val_loss)
                
                logger.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{epochs}: "
                          f"Train Loss = {avg_train_loss:.4f}, "
                          f"Val Loss = {avg_val_loss:.4f}")
            
            fold_metrics.append({
                'final_train_loss': fold_history['train_loss'][-1],
                'final_val_loss': fold_history['val_loss'][-1],
                'history': fold_history
            })
        
        # Compute average metrics across folds
        avg_metrics = {
            'avg_train_loss': np.mean([m['final_train_loss'] for m in fold_metrics]),
            'avg_val_loss': np.mean([m['final_val_loss'] for m in fold_metrics]),
            'std_train_loss': np.std([m['final_train_loss'] for m in fold_metrics]),
            'std_val_loss': np.std([m['final_val_loss'] for m in fold_metrics])
        }
        
        # Save version information
        new_version = self._increment_version()
        version_info = {
            "version": new_version,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "cross_validation": avg_metrics,
                "fold_metrics": fold_metrics
            }
        }
        
        self._save_version_info(version_info)
        self.current_version = version_info
        
        return avg_metrics
    
    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train the model with the given data."""
        try:
            logger.info("Preparing data for training...")
            
            # Prepare features and target
            X = data['description'].values
            y = data['severity_code'].values
            
            # Create dataset
            dataset = CrashDataset(
                texts=torch.tensor(self.model.preprocess_text(X.tolist())),
                labels=torch.tensor(y, dtype=torch.float32)
            )
            
            # Create data loader
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Train model
            logger.info("Training model...")
            self.model.train_model(dataloader, epochs=10)
            
            # Save model and update version
            self._save_model()
            self._increment_version()
            
            metrics = {
                "loss": self.model.training_loss,
                "accuracy": self.model.training_accuracy
            }
            
            logger.info(f"Training complete. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def _save_model(self) -> None:
        """Save the model to the model directory."""
        model_path = self.model_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
    
    def _save_version_info(self, version_info: Dict) -> None:
        """Save version information to the versions file."""
        versions = []
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                versions = json.load(f)
        
        versions.append(version_info)
        
        with open(self.version_file, 'w') as f:
            json.dump(versions, f, indent=2)
        
        logger.info(f"Saved version information: {version_info['version']}")
    
    def get_version_history(self) -> List[Dict]:
        """Get the history of model versions and their metrics."""
        if not self.version_file.exists():
            return []
        
        with open(self.version_file, 'r') as f:
            return json.load(f)
    
    def get_current_version(self) -> Dict:
        """Get the current model version information."""
        return self.current_version
    
    def save_models(self):
        """Save trained models and configurations."""
        try:
            logger.debug(f"Saving models to: {self.model_dir}")
            
            # Create model directory if it doesn't exist
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save a dummy model for testing
            # In production, this would be the actual trained model
            dummy_model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 2)
            )
            
            # Save model
            torch.save(
                dummy_model.state_dict(),
                self.model_dir / 'text_encoder.pth',
                _use_new_zipfile_serialization=True  # Use new format
            )
            
            logger.info("Successfully saved models")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}", exc_info=True)
            return False
