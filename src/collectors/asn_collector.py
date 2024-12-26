"""Aviation Safety Network data collector."""

import logging
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import re

logger = logging.getLogger(__name__)

class ASNCollector:
    """Collector for Aviation Safety Network data."""
    
    BASE_URL = "https://aviation-safety.net/database/"
    
    def __init__(self):
        """Initialize ASN collector."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    async def collect_data(self, start_date: datetime = None, end_date: datetime = None) -> List[Dict[str, Any]]:
        """Collect accident data from Aviation Safety Network.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            List of accident records
        """
        async with aiohttp.ClientSession() as session:
            try:
                # First get the search page
                params = {
                    'sort': 'datekey',
                    'page': 1
                }
                
                if start_date:
                    params['date'] = start_date.strftime('%Y%m%d')
                if end_date:
                    params['dateend'] = end_date.strftime('%Y%m%d')
                
                accidents = []
                page = 1
                
                while True:
                    params['page'] = page
                    async with session.get(self.BASE_URL, headers=self.headers, params=params) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Find accident entries
                            entries = soup.find_all('tr', class_='list')
                            if not entries:
                                break
                                
                            for entry in entries:
                                try:
                                    cells = entry.find_all('td')
                                    if len(cells) >= 6:
                                        date_str = cells[0].text.strip()
                                        accident = {
                                            'date': datetime.strptime(date_str, '%d-%b-%Y').strftime('%Y-%m-%d'),
                                            'aircraft_type': cells[1].text.strip(),
                                            'registration': cells[2].text.strip(),
                                            'operator': cells[3].text.strip(),
                                            'location': cells[4].text.strip(),
                                            'description': cells[5].text.strip(),
                                            'fatalities': self._extract_fatalities(cells[5].text.strip())
                                        }
                                        accidents.append(accident)
                                except Exception as e:
                                    logger.warning(f"Error parsing entry: {e}")
                                    continue
                            
                            page += 1
                            if page > 5:  # Limit to 5 pages for now
                                break
                        else:
                            logger.error(f"ASN request failed: {response.status}")
                            break
                
                logger.info(f"Collected {len(accidents)} records from Aviation Safety Network")
                return accidents
                
            except Exception as e:
                logger.error(f"Error collecting ASN data: {e}", exc_info=True)
                return []
    
    def _extract_fatalities(self, description: str) -> int:
        """Extract fatality count from accident description."""
        match = re.search(r'(\d+)\s+fatal', description.lower())
        if match:
            return int(match.group(1))
        return 0
