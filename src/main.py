import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional, List
import pandas as pd

from collectors.crash_data_collector import CrashDataCollector
from models.ai_model import AIModel
from analysis.data_analyzer import DataAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrashAIApp:
    """Main application class for the Crash AI system."""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.collector = CrashDataCollector(data_dir=str(self.data_dir))
        self.model = AIModel(model_dir=str(self.model_dir))
        self.analyzer = DataAnalyzer(data_dir=str(self.data_dir))
    
    async def collect_data(self) -> None:
        """Collect new crash data."""
        logger.info("Starting data collection...")
        async with self.collector as collector:
            await collector.collect_data()
        logger.info("Data collection completed")
    
    def train_model(self) -> None:
        """Train the AI model on collected data."""
        logger.info("Loading latest crash data...")
        data = self.analyzer.load_data()
        if data.empty:
            logger.error("No training data found")
            return
        
        logger.info("Training model...")
        history = self.model.train(
            train_data=data,
            text_col='description',
            target_cols=['severity', 'risk_factor'],
            epochs=10
        )
        logger.info(f"Training completed. Final loss: {history['loss'][-1]:.4f}")
        
        logger.info("Saving model...")
        self.model.save_models()
    
    def analyze_patterns(self) -> None:
        """Analyze crash patterns in the collected data."""
        logger.info("Loading data for analysis...")
        data = self.analyzer.load_data()
        if data.empty:
            logger.error("No data found for analysis")
            return
        
        logger.info("Analyzing patterns...")
        patterns = self.analyzer.analyze_patterns()
        
        logger.info("Generating visualizations...")
        self.analyzer.generate_visualizations(str(self.data_dir / "figures"))
        logger.info(f"Visualizations saved to {self.data_dir / 'figures'}")
        
        # Print key insights
        logger.info("\nKey Insights:")
        for pattern_type, results in patterns.items():
            logger.info(f"\n{pattern_type.replace('_', ' ').title()}:")
            if isinstance(results, pd.DataFrame):
                logger.info(results.to_string())
            elif isinstance(results, dict):
                for key, value in results.items():
                    logger.info(f"{key}: {value}")
    
    def predict_risk(self, descriptions: List[str]) -> None:
        """Make predictions for new crash scenarios."""
        logger.info("Loading model...")
        try:
            self.model.load_models()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
        
        logger.info("Making predictions...")
        predictions = self.model.predict(
            texts=descriptions,
            target_cols=['severity', 'risk_factor']
        )
        
        logger.info("\nPredictions:")
        for desc, pred in zip(descriptions, predictions.itertuples()):
            logger.info(f"\nScenario: {desc}")
            logger.info(f"Severity: {pred.severity:.2f}")
            logger.info(f"Risk Factor: {pred.risk_factor:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Crash AI Analysis System")
    parser.add_argument('action', choices=['collect', 'train', 'analyze', 'predict'],
                      help='Action to perform')
    parser.add_argument('--descriptions', nargs='+',
                      help='Descriptions for prediction (only used with predict action)')
    
    args = parser.parse_args()
    app = CrashAIApp()
    
    if args.action == 'collect':
        asyncio.run(app.collect_data())
    elif args.action == 'train':
        app.train_model()
    elif args.action == 'analyze':
        app.analyze_patterns()
    elif args.action == 'predict':
        if not args.descriptions:
            logger.error("Please provide descriptions for prediction")
            return
        app.predict_risk(args.descriptions)

if __name__ == "__main__":
    main()
