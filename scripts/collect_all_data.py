"""Script to collect data from all aviation safety sources."""

import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from src.collectors.crash_data_collector import CrashDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def collect_data():
    """Collect data from all sources."""
    # Initialize collector
    collector = CrashDataCollector()
    
    # Set date range for last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)
    
    logger.info(f"Collecting data from {start_date.date()} to {end_date.date()}")
    
    # Collect from all sources
    results = await collector.collect_all_sources(start_date, end_date)
    
    # Print summary
    logger.info("\nCollection Summary:")
    for source, df in results.items():
        logger.info(f"- {source}: {len(df)} records")
    
    # Show available sources in MongoDB
    sources = collector.get_available_sources()
    logger.info("\nAvailable Sources in MongoDB:")
    for source in sources:
        logger.info(
            f"- {source['_id']}: {source['count']} records "
            f"({source['earliest_date'].strftime('%Y-%m-%d')} to {source['latest_date'].strftime('%Y-%m-%d')})"
        )

if __name__ == "__main__":
    asyncio.run(collect_data())
