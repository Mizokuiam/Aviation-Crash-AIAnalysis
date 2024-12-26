"""FAA Incident Data collector."""

import logging
from datetime import datetime
import aiohttp
from typing import List, Dict, Any
import pandas as pd
import io

logger = logging.getLogger(__name__)

class FAACollector:
    """Collector for FAA incident data."""
    
    BASE_URL = "https://www.faa.gov/data_research/accident_incident/preliminary_data"
    
    def __init__(self):
        """Initialize FAA collector."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json,application/xml,text/csv',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    async def collect_data(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """Collect incident data from FAA.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            List of incident records
        """
        async with aiohttp.ClientSession() as session:
            try:
                # FAA provides monthly data files
                current_year = datetime.now().year
                incidents = []
                
                # Download last 12 months of data
                for month in range(1, 13):
                    url = f"{self.BASE_URL}/incidents_{current_year}_{month:02d}.csv"
                    
                    async with session.get(url, headers=self.headers) as response:
                        if response.status == 200:
                            csv_data = await response.text()
                            df = pd.read_csv(io.StringIO(csv_data))
                            
                            for _, row in df.iterrows():
                                incident = {
                                    'date': row.get('Event_Date', ''),
                                    'location': f"{row.get('City', '')} {row.get('State', '')}",
                                    'aircraft_type': row.get('Aircraft_Type', ''),
                                    'severity': row.get('Severity', 'Incident'),
                                    'description': row.get('Description', ''),
                                    'flight_phase': row.get('Flight_Phase', ''),
                                    'weather': row.get('Weather_Condition', ''),
                                    'airport': row.get('Airport_Name', '')
                                }
                                
                                # Apply date filters if provided
                                incident_date = datetime.strptime(incident['date'], '%Y-%m-%d')
                                if start_date and incident_date < start_date:
                                    continue
                                if end_date and incident_date > end_date:
                                    continue
                                    
                                incidents.append(incident)
                        else:
                            logger.warning(f"Could not download FAA data for {current_year}-{month:02d}")
                
                logger.info(f"Collected {len(incidents)} records from FAA")
                return incidents
                
            except Exception as e:
                logger.error(f"Error collecting FAA data: {e}", exc_info=True)
                return []
