import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from collectors.crash_data_collector import CrashDataCollector
from models.ai_model import AIModel
from models.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomationMonitor:
    """Monitor class for automated data collection and model retraining."""
    
    def __init__(self, 
                 data_dir: str = "data",
                 model_dir: str = "models",
                 config_file: str = "automation_config.json"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.config_file = Path(config_file)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.collector = CrashDataCollector(data_dir=str(self.data_dir / "raw"))
        self.model = AIModel(model_dir=str(self.model_dir))
        self.trainer = ModelTrainer(model_dir=str(self.model_dir))
        
        # URLs for crash data collection from NTSB API
        self.default_urls = [
            "https://data.ntsb.gov/carol-main-public/basic-search",  # Main search page
            "https://data.ntsb.gov/carol-main-public/api/Query/GetCountsByYear",  # Yearly stats
            "https://data.ntsb.gov/carol-main-public/api/Query/GetAccidentList"   # Detailed records
        ]
        
        self.config = self._load_config()
        self._ensure_initial_setup()
    
    def _ensure_initial_setup(self) -> None:
        """Ensure initial model files exist."""
        vocab_file = self.model_dir / "vocab.pkl"
        if not vocab_file.exists():
            logger.info("Creating initial model files...")
            # Create a minimal vocabulary
            vocab = {"<PAD>": 0, "<UNK>": 1}
            with open(vocab_file, 'wb') as f:
                pickle.dump(vocab, f)
            
            # Save initial model configuration
            config = {
                "vocab_size": len(vocab),
                "embedding_dim": 100,
                "hidden_dim": 128,
                "output_dim": 2  # severity and risk_factor
            }
            with open(self.model_dir / "config.pkl", 'wb') as f:
                pickle.dump(config, f)
            
            logger.info("Initial model files created")
    
    def _load_config(self) -> dict:
        """Load automation configuration."""
        default_config = {
            "collection_interval_hours": 24,
            "retraining_interval_days": 7,
            "min_new_samples": 100,
            "last_collection": None,
            "last_training": None
        }
        
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        return default_config
    
    def _save_config(self) -> None:
        """Save current configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    async def _collect_data(self) -> bool:
        """Collect new data and return True if collection was performed."""
        now = datetime.now()
        last_collection = (datetime.fromisoformat(self.config["last_collection"])
                         if self.config["last_collection"] else None)
        
        if (not last_collection or 
            now - last_collection > timedelta(hours=self.config["collection_interval_hours"])):
            logger.info("Starting automated data collection...")
            
            try:
                df = await self.collector.collect_data(self.default_urls)
                logger.info(f"Collected {len(df)} records")
                
                self.config["last_collection"] = now.isoformat()
                self._save_config()
                return True
            except Exception as e:
                logger.error(f"Error during data collection: {e}")
                return False
        
        return False
    
    def _should_retrain(self) -> bool:
        """Check if model retraining is needed."""
        now = datetime.now()
        last_training = (datetime.fromisoformat(self.config["last_training"])
                        if self.config["last_training"] else None)
        
        if not last_training:
            return True
        
        if now - last_training > timedelta(days=self.config["retraining_interval_days"]):
            # Check if we have enough new data
            try:
                data = self.model.load_latest_data()
                if len(data) >= self.config["min_new_samples"]:
                    return True
            except Exception as e:
                logger.error(f"Error checking data for retraining: {e}")
        
        return False
    
    def _preprocess_data(self, data):
        """Preprocess raw crash data."""
        try:
            # Create processed directory if it doesn't exist
            processed_dir = self.data_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            
            # Basic preprocessing
            data['severity_code'] = data['severity'].map({
                'Minor': 0,
                'Moderate': 1,
                'Severe': 2
            })
            
            # Save processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = processed_dir / f"crash_data_{timestamp}.csv"
            data.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None

    async def _retrain_model(self) -> None:
        """Retrain the model with latest data."""
        try:
            logger.info("Starting automated model retraining...")
            
            # Load and preprocess latest data
            raw_data = self.collector.load_data()
            if raw_data.empty:
                logger.warning("No raw data available for training")
                return
                
            processed_data = self._preprocess_data(raw_data)
            if processed_data is None:
                logger.error("Failed to preprocess data")
                return
            
            # Train model
            self.trainer.train(processed_data)
            logger.info("Model retraining complete")
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
    
    async def run(self) -> None:
        """Run the automation monitor."""
        logger.info("Starting automation monitor...")
        
        while True:
            try:
                # Collect new data
                data_collected = await self._collect_data()
                if data_collected:
                    logger.info("New data collected successfully")
                
                # Retrain model if needed
                await self._retrain_model()
                logger.info("Model retrained successfully")
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in automation monitor: {e}")
                await asyncio.sleep(3600)  # Wait an hour before retrying

async def main():
    monitor = AutomationMonitor()
    await monitor.run()

if __name__ == "__main__":
    asyncio.run(main())
