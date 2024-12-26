"""MongoDB service for crash data storage and retrieval."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING, IndexModel
from pymongo.database import Database
from pymongo.collection import Collection
from cachetools import TTLCache, cached
import functools

logger = logging.getLogger(__name__)

class MongoDBService:
    """Service for MongoDB operations."""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", database_name: str = "aviation_crashes"):
        """Initialize MongoDB service with connection details."""
        self.client = MongoClient(connection_string)
        self.db: Database = self.client[database_name]
        self.crashes: Collection = self.db.crashes
        self.metadata: Collection = self.db.metadata
        self._setup_indexes()
        
        # Initialize caches
        self.source_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes TTL
        self.data_cache = TTLCache(maxsize=1000, ttl=60)    # 1 minute TTL
        self.stats_cache = TTLCache(maxsize=100, ttl=300)   # 5 minutes TTL
    
    def _setup_indexes(self):
        """Set up database indexes for better query performance."""
        try:
            # Create indexes for crash_data collection
            crash_indexes = [
                IndexModel([("date", ASCENDING)]),
                IndexModel([("source", ASCENDING)]),
                IndexModel([("severity", ASCENDING)]),
                IndexModel([("aircraft_type", ASCENDING)]),
                IndexModel([("operator", ASCENDING)]),
                IndexModel([("description", "text")])    # Text search index
            ]
            self.crashes.create_indexes(crash_indexes)
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}", exc_info=True)
    
    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def get_available_sources(self) -> List[Dict[str, Any]]:
        """Get available data sources with caching."""
        pipeline = [
            {
                "$group": {
                    "_id": "$source",
                    "count": {"$sum": 1},
                    "earliest_date": {"$min": "$date"},
                    "latest_date": {"$max": "$date"},
                    "last_import": {"$max": "$import_timestamp"}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "count": 1,
                    "date_range": {
                        "$concat": [
                            {"$dateToString": {"format": "%Y-%m-%d", "date": "$earliest_date"}},
                            " to ",
                            {"$dateToString": {"format": "%Y-%m-%d", "date": "$latest_date"}}
                        ]
                    },
                    "last_import": 1
                }
            }
        ]
        
        try:
            return list(self.crashes.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Error getting available sources: {e}", exc_info=True)
            return []
    
    def _get_cache_key(self, sources: List[str], start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None, filters: Optional[Dict] = None) -> str:
        """Generate cache key for data queries."""
        key_parts = [
            ",".join(sorted(sources)),
            start_date.isoformat() if start_date else "None",
            end_date.isoformat() if end_date else "None",
            str(sorted(filters.items())) if filters else "None"
        ]
        return "|".join(key_parts)
    
    def get_data(self, sources: List[str], start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None, filters: Optional[Dict] = None,
                 page: int = 1, page_size: int = 100) -> Tuple[List[Dict], int]:
        """Get crash data with caching and pagination."""
        cache_key = self._get_cache_key(sources, start_date, end_date, filters)
        
        # Try to get from cache
        if cache_key in self.data_cache:
            data = self.data_cache[cache_key]
            total = len(data)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            return data[start_idx:end_idx], total
        
        # Build query
        query = {"source": {"$in": sources}}
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = start_date
            if end_date:
                date_query["$lte"] = end_date
            query["date"] = date_query
        
        if filters:
            query.update(filters)
        
        try:
            # Use aggregation for better performance
            pipeline = [
                {"$match": query},
                {"$sort": {"date": -1}},
                {
                    "$project": {
                        "_id": 0,
                        "date": 1,
                        "location": 1,
                        "severity": 1,
                        "aircraft_type": 1,
                        "operator": 1,
                        "description": 1,
                        "source": 1
                    }
                }
            ]
            
            # Execute query
            cursor = self.crashes.aggregate(pipeline)
            data = list(cursor)
            
            # Cache results
            self.data_cache[cache_key] = data
            
            # Return paginated results
            total = len(data)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            return data[start_idx:end_idx], total
            
        except Exception as e:
            logger.error(f"Error getting crash data: {e}", exc_info=True)
            return [], 0
    
    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def get_statistics(self, sources: List[str]) -> Dict[str, Any]:
        """Get statistics with caching."""
        try:
            pipeline = [
                {"$match": {"source": {"$in": sources}}},
                {
                    "$group": {
                        "_id": None,
                        "total_incidents": {"$sum": 1},
                        "severity_counts": {
                            "$push": "$severity"
                        },
                        "aircraft_types": {
                            "$addToSet": "$aircraft_type"
                        },
                        "operators": {
                            "$addToSet": "$operator"
                        }
                    }
                }
            ]
            
            result = next(self.crashes.aggregate(pipeline), None)
            if not result:
                return {}
            
            # Process severity counts
            severity_counts = {
                severity: result["severity_counts"].count(severity)
                for severity in set(result["severity_counts"])
            }
            
            return {
                "total_incidents": result["total_incidents"],
                "severity_distribution": severity_counts,
                "unique_aircraft_types": len(result["aircraft_types"]),
                "unique_operators": len(result["operators"])
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}", exc_info=True)
            return {}
    
    def search_incidents(self, query: str, sources: List[str]) -> List[Dict]:
        """Perform text search on incidents."""
        try:
            pipeline = [
                {
                    "$match": {
                        "$and": [
                            {"source": {"$in": sources}},
                            {"$text": {"$search": query}}
                        ]
                    }
                },
                {"$sort": {"score": {"$meta": "textScore"}}},
                {"$limit": 100},
                {
                    "$project": {
                        "_id": 0,
                        "date": 1,
                        "location": 1,
                        "severity": 1,
                        "aircraft_type": 1,
                        "operator": 1,
                        "description": 1,
                        "score": {"$meta": "textScore"}
                    }
                }
            ]
            
            return list(self.crashes.aggregate(pipeline))
            
        except Exception as e:
            logger.error(f"Error searching incidents: {e}", exc_info=True)
            return []
    
    def clear_caches(self):
        """Clear all caches."""
        self.source_cache.clear()
        self.data_cache.clear()
        self.stats_cache.clear()
    
    def _standardize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize record fields for consistency across sources.
        
        Args:
            record: Raw record from source
            
        Returns:
            Standardized record
        """
        # Convert date string to datetime if needed
        if isinstance(record.get('date'), str):
            try:
                record['date'] = datetime.strptime(record['date'], '%Y-%m-%d')
            except ValueError:
                logger.warning(f"Invalid date format in record: {record.get('date')}")
                record['date'] = None
        
        # Standardize severity field
        severity = record.get('severity', '').lower()
        if 'fatal' in severity:
            record['severity'] = 'Fatal'
        elif 'incident' in severity:
            record['severity'] = 'Incident'
        elif severity:
            record['severity'] = severity.title()
        else:
            record['severity'] = 'Unknown'
        
        # Standardize aircraft type
        aircraft_type = record.get('aircraft_type', '')
        if aircraft_type:
            record['aircraft_type'] = aircraft_type.strip()
        else:
            record['aircraft_type'] = 'Unknown'
        
        # Ensure all required fields exist
        required_fields = {
            'location': 'Unknown',
            'description': '',
            'flight_phase': 'Unknown',
            'weather': 'Unknown'
        }
        
        for field, default in required_fields.items():
            if field not in record or not record[field]:
                record[field] = default
        
        return record
    
    def store_crashes(self, crashes: List[Dict[str, Any]], source: str) -> int:
        """Store crash records in MongoDB.
        
        Args:
            crashes: List of crash records
            source: Name of the data source
            
        Returns:
            Number of records stored
        """
        if not crashes:
            logger.warning("No crashes to store")
            return 0
        
        try:
            # Add source and standardize records
            standardized = []
            for crash in crashes:
                crash['source'] = source
                standardized.append(self._standardize_record(crash))
            
            # Insert records
            result = self.crashes.insert_many(standardized)
            stored = len(result.inserted_ids)
            
            # Update metadata with last import time
            self.metadata.update_one(
                {'source': source},
                {
                    '$set': {
                        'last_import': datetime.now(),
                        'record_count': stored
                    }
                },
                upsert=True
            )
            
            logger.info(f"Stored {stored} records from {source}")
            return stored
            
        except Exception as e:
            logger.error(f"Error storing crashes: {e}", exc_info=True)
            return 0
    
    def get_crashes(self, source: str = None, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get crash records from MongoDB.
        
        Args:
            source: Optional source filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame containing crash records
        """
        try:
            # Build query
            query = {}
            if source:
                query['source'] = source
            if start_date or end_date:
                query['date'] = {}
                if start_date:
                    query['date']['$gte'] = start_date
                if end_date:
                    query['date']['$lte'] = end_date
            
            # Get records
            records = list(self.crashes.find(query))
            
            if not records:
                logger.warning(f"No records found for query: {query}")
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=[
                    'date', 'location', 'aircraft_type', 'severity',
                    'description', 'flight_phase', 'weather', 'source'
                ])
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Drop MongoDB _id column
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Ensure all required columns exist
            required_columns = [
                'date', 'location', 'aircraft_type', 'severity',
                'description', 'flight_phase', 'weather', 'source'
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    if col == 'date':
                        df[col] = pd.NaT
                    else:
                        df[col] = 'Unknown'
            
            # Convert date to datetime if needed
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting crashes: {e}", exc_info=True)
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'date', 'location', 'aircraft_type', 'severity',
                'description', 'flight_phase', 'weather', 'source'
            ])
    
    def get_source_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for each data source.
        
        Returns:
            List of dictionaries containing source statistics
        """
        try:
            pipeline = [
                {
                    '$group': {
                        '_id': '$source',
                        'count': {'$sum': 1},
                        'earliest_date': {'$min': '$date'},
                        'latest_date': {'$max': '$date'}
                    }
                }
            ]
            
            stats = list(self.crashes.aggregate(pipeline))
            
            # Add last import times from metadata
            for stat in stats:
                source = stat['_id']
                metadata = self.metadata.find_one({'source': source})
                if metadata and 'last_import' in metadata:
                    stat['last_import'] = metadata['last_import']
                else:
                    stat['last_import'] = None
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting source stats: {e}", exc_info=True)
            return []
