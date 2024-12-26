"""Script to load sample crash data into MongoDB."""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

# Add parent directory to path for imports
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.database.mongodb_service import MongoDBService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load sample data into MongoDB."""
    # Initialize MongoDB service
    db = MongoDBService()
    
    # Sample data files for each source
    sample_files = {
        'NTSB': 'ntsb_sample_data.csv',
        'Aviation Safety Network': 'asn_sample_data.csv',
        'FAA': 'faa_sample_data.csv'
    }
    
    for source, filename in sample_files.items():
        # Load sample data from CSV
        sample_file = Path(root_dir) / "data" / "raw" / filename
        if not sample_file.exists():
            logger.error(f"Sample data file not found: {sample_file}")
            continue
            
        # Read sample data
        df = pd.read_csv(sample_file)
        logger.info(f"Loaded {len(df)} records from {source} sample data")
        
        # Convert to list of dictionaries
        crashes = df.to_dict('records')
        
        # Store in MongoDB
        stored = db.store_crashes(crashes, source)
        logger.info(f"Stored {stored} records from {source} in MongoDB")
    
    # Show available sources
    sources = db.get_source_stats()
    logger.info("\nAvailable Sources:")
    for source in sources:
        logger.info(
            f"- {source['_id']}: {source['count']} records "
            f"({source['earliest_date'].strftime('%Y-%m-%d')} to {source['latest_date'].strftime('%Y-%m-%d')})"
        )

if __name__ == "__main__":
    load_sample_data()
