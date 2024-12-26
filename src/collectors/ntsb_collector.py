"""NTSB Aviation Accident Database collector."""

import logging
from datetime import datetime, timedelta
import aiohttp
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

class NTSBCollector:
    """Collector for NTSB aviation accident data."""
    
    BASE_URL = "https://data.ntsb.gov/carol-main-public/api/Query/GetAccidentList"
    
    def __init__(self):
        """Initialize NTSB collector."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    async def collect_data(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """Collect accident data from NTSB API.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            List of accident records
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)  # Last year
        if not end_date:
            end_date = datetime.now()
            
        params = {
            'eventStartDate': start_date.strftime('%Y-%m-%d'),
            'eventEndDate': end_date.strftime('%Y-%m-%d'),
            'eventType': 'Accident',
            'eventCountry': 'United States'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.BASE_URL, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        accidents = []
                        
                        for record in data.get('results', []):
                            accident = {
                                'date': record.get('eventDate'),
                                'location': f"{record.get('city', '')} {record.get('state', '')}",
                                'aircraft_type': record.get('aircraftCategory', ''),
                                'severity': record.get('injurySeverity', 'Unknown'),
                                'description': record.get('narrative', ''),
                                'registration': record.get('registration', ''),
                                'operator': record.get('operator', ''),
                                'flight_purpose': record.get('flightPurpose', ''),
                                'weather_condition': record.get('weatherCondition', ''),
                                'broad_phase': record.get('broadPhaseOfFlight', '')
                            }
                            accidents.append(accident)
                            
                        logger.info(f"Collected {len(accidents)} records from NTSB")
                        return accidents
                    else:
                        logger.error(f"NTSB API error: {response.status}")
                        return []
                        
            except Exception as e:
                logger.error(f"Error collecting NTSB data: {e}", exc_info=True)
                return []
