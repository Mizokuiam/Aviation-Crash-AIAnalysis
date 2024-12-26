import aiohttp
import asyncio
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
from src.database.mongodb_service import MongoDBService
from src.collectors.ntsb_collector import NTSBCollector
from src.collectors.asn_collector import ASNCollector
from src.collectors.faa_collector import FAACollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrashDataCollector:
    """Collector for aviation crash data from multiple sources."""
    
    def __init__(self, data_dir: str = "data/raw", mongo_uri: str = "mongodb://localhost:27017/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = MongoDBService(mongo_uri)
        
        # Initialize specialized collectors
        self.ntsb_collector = NTSBCollector()
        self.asn_collector = ASNCollector()
        self.faa_collector = FAACollector()
        
    async def collect_from_source(self, source: str, start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """Collect data from a specific source.
        
        Args:
            source: Name of the data source
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of crash/incident records
        """
        try:
            if source == "NTSB":
                return await self.ntsb_collector.collect_data(start_date, end_date)
            elif source == "Aviation Safety Network":
                return await self.asn_collector.collect_data(start_date, end_date)
            elif source == "FAA":
                return await self.faa_collector.collect_data(start_date, end_date)
            else:
                logger.warning(f"Unknown source: {source}")
                return []
                
        except Exception as e:
            logger.error(f"Error collecting from {source}: {e}", exc_info=True)
            return []
            
    async def collect_all_sources(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, pd.DataFrame]:
        """Collect data from all sources.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary mapping source names to their respective DataFrames
        """
        sources = ["NTSB", "Aviation Safety Network", "FAA"]
        results = {}
        
        for source in sources:
            try:
                logger.info(f"Collecting data from {source}...")
                records = await self.collect_from_source(source, start_date, end_date)
                
                if records:
                    # Store in MongoDB
                    stored = self.db.store_crashes(records, source)
                    logger.info(f"Stored {stored} records from {source} in MongoDB")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(records)
                    results[source] = df
                    
                    # Save backup CSV
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = self.data_dir / f"crash_data_{source.lower().replace(' ', '_')}_{timestamp}.csv"
                    df.to_csv(output_file, index=False)
                    logger.info(f"Saved backup to {output_file}")
                    
            except Exception as e:
                logger.error(f"Error processing {source} data: {e}", exc_info=True)
                
        return results
    
    def get_source_data(self, source: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get crash data for a specific source from MongoDB.
        
        Args:
            source: Name of the data source
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame containing crash data
        """
        return self.db.get_crashes(source, start_date, end_date)
    
    def get_available_sources(self) -> List[Dict[str, Any]]:
        """Get list of available data sources with statistics.
        
        Returns:
            List of dictionaries containing source information
        """
        return self.db.get_source_stats()

    def get_data(self, sources: List[str], start_date: Optional[Union[str, datetime]] = None, 
                 end_date: Optional[Union[str, datetime]] = None, filters: Optional[Dict] = None) -> pd.DataFrame:
        """Get crash data from MongoDB with optional filtering.
        
        Args:
            sources: List of data sources to include
            start_date: Optional start date filter (string or datetime)
            end_date: Optional end date filter (string or datetime)
            filters: Optional dictionary of additional filters
            
        Returns:
            DataFrame containing crash records
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            except (ValueError, TypeError):
                start_date = None
                
        if isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            except (ValueError, TypeError):
                end_date = None

        data, _ = self.db.get_data(sources, start_date, end_date, filters)
        df = pd.DataFrame(data) if data else pd.DataFrame()
        
        # Handle missing values in operator column
        if 'operator' in df.columns:
            df['operator'] = df['operator'].fillna('Unknown')
            
        return df

    def load_data(self) -> pd.DataFrame:
        """Load the latest crash data from file."""
        try:
            # First try to load sample data
            sample_file = self.data_dir / "sample_crash_data.csv"
            if sample_file.exists():
                logger.info(f"Loading sample data from {sample_file}")
                data = pd.read_csv(sample_file)
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                return data
            
            # If no sample data, try to load collected data
            files = list(self.data_dir.glob("crash_data_*.csv"))
            if not files:
                logger.warning("No data files found")
                return pd.DataFrame()
            
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading data from {latest_file}")
            
            data = pd.read_csv(latest_file)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def load_latest_data(self) -> pd.DataFrame:
        """Load the most recent crash data file."""
        try:
            # Get list of all data files
            data_files = list(Path(self.data_dir).glob("crash_data_*.csv"))
            if not data_files:
                # If no data files exist, use sample data
                sample_file = Path(self.data_dir) / "sample_crash_data.csv"
                if sample_file.exists():
                    return pd.read_csv(sample_file)
                else:
                    logger.warning("No sample data found")
                    return pd.DataFrame()
            
            # Get most recent file
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading data from {latest_file}")
            return pd.read_csv(latest_file)
            
        except Exception as e:
            logger.error(f"Error loading latest data: {e}", exc_info=True)
            return pd.DataFrame()

# Example usage
async def main():
    collector = CrashDataCollector()
    results = await collector.collect_all_sources()
    print("Data collection completed")
