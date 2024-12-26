"""Configuration for aviation crash data sources."""

DATA_SOURCES = {
    'NTSB Database': {
        'url': 'https://data.ntsb.gov/carol-main-public/api/Query/GetAccidentList',
        'description': 'Official NTSB Aviation Accident Database',
        'type': 'api'
    },
    'Aviation Safety Network': {
        'url': 'https://aviation-safety.net/database/',
        'description': 'Comprehensive aviation accident database',
        'type': 'web'
    },
    'ICAO Safety Report': {
        'url': 'https://www.icao.int/safety/Pages/Safety-Report.aspx',
        'description': 'ICAO annual safety reports',
        'type': 'report'
    },
    'Sample Data': {
        'url': 'sample',
        'description': 'Local sample dataset for testing',
        'type': 'local'
    }
}

def get_data_sources():
    """Get list of available data sources."""
    return {
        name: {
            'url': info['url'],
            'description': info['description'],
            'type': info['type']
        }
        for name, info in DATA_SOURCES.items()
    }

def get_source_url(source_name: str) -> str:
    """Get URL for a specific data source."""
    if source_name in DATA_SOURCES:
        return DATA_SOURCES[source_name]['url']
    return None
