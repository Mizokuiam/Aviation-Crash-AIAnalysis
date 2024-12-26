import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import spacy
from collections import Counter
import logging
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Analyzes crash data and extracts insights."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        logger.debug(f"Initializing DataAnalyzer with data directory: {self.data_dir}")
        self.nlp = spacy.load('en_core_web_sm')
        self.crash_data = None
        
        # Ensure data directory exists
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")
        
        # Load data on initialization
        self.load_data()
        if self.crash_data is not None:
            logger.info(f"Loaded {len(self.crash_data)} records on initialization")
        else:
            logger.warning("No data loaded on initialization")
        
    def load_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Load crash data from file."""
        try:
            logger.debug(f"Looking for data in directory: {self.data_dir}")
            
            # Try loading sample data first
            sample_file = self.data_dir / "sample_crash_data.csv"
            if sample_file.exists():
                logger.debug(f"Found sample data file: {sample_file}")
                # Read the CSV with explicit column types
                self.crash_data = pd.read_csv(
                    sample_file,
                    dtype={
                        'date': str,
                        'location': str,
                        'aircraft_type': str,
                        'severity': str,
                        'description': str
                    }
                )
                logger.debug(f"Loaded sample data shape: {self.crash_data.shape}")
                logger.debug(f"Sample data columns: {self.crash_data.columns.tolist()}")
                
                # Convert date column to datetime
                if 'date' in self.crash_data.columns:
                    logger.debug("Converting date column to datetime")
                    try:
                        self.crash_data['date'] = pd.to_datetime(self.crash_data['date'], format='%Y-%m-%d')
                    except Exception as e:
                        logger.error(f"Error converting dates: {e}")
                        return pd.DataFrame()
                
                return self.crash_data
            
            # If no sample data, look for other data files
            if filename:
                path = self.data_dir / filename
                logger.debug(f"Looking for specific file: {path}")
            else:
                files = list(self.data_dir.glob("crash_data_*.csv"))
                if not files:
                    logger.warning(f"No crash data files found in {self.data_dir}")
                    return pd.DataFrame()
                path = max(files, key=lambda x: x.stat().st_mtime)
                logger.debug(f"Using latest file: {path}")
            
            # Read the CSV with explicit column types
            self.crash_data = pd.read_csv(
                path,
                dtype={
                    'date': str,
                    'location': str,
                    'aircraft_type': str,
                    'severity': str,
                    'description': str
                }
            )
            logger.debug(f"Loaded data shape: {self.crash_data.shape}")
            
            # Convert date column to datetime
            if 'date' in self.crash_data.columns:
                try:
                    self.crash_data['date'] = pd.to_datetime(self.crash_data['date'], format='%Y-%m-%d')
                except Exception as e:
                    logger.error(f"Error converting dates: {e}")
                    return pd.DataFrame()
            
            return self.crash_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return pd.DataFrame()
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in crash data."""
        if self.crash_data is None:
            logger.warning("No data loaded. Call load_data() first.")
            return {}
        
        try:
            logger.debug("Starting pattern analysis")
            results = {
                'temporal_patterns': self._analyze_temporal_patterns(),
                'geographical_patterns': self._analyze_geographical_patterns(),
                'causal_factors': self._analyze_causal_factors(),
                'severity_analysis': self._analyze_severity(),
                'text_insights': self._analyze_text_data()
            }
            logger.debug("Pattern analysis complete")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}", exc_info=True)
            return {}
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in crash data."""
        try:
            if self.crash_data is None or self.crash_data.empty:
                logger.warning("No data available for temporal analysis")
                return {}
            
            # Convert date strings to datetime objects
            dates = pd.to_datetime(self.crash_data['date'])
            
            # Group by month and count crashes
            monthly = pd.DataFrame({
                'dates': dates,
                'counts': [1] * len(dates)  # One count per crash
            })
            monthly = monthly.set_index('dates')
            monthly = monthly.resample('M').sum()
            
            # Calculate 3-month moving average
            moving_avg = monthly['counts'].rolling(window=3, min_periods=1).mean()
            
            # Reset index to get dates as column
            monthly = monthly.reset_index()
            
            return {
                'dates': monthly['dates'].tolist(),
                'counts': monthly['counts'].tolist(),
                'moving_avg': moving_avg.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}", exc_info=True)
            return {}
    
    def _analyze_geographical_patterns(self) -> Dict[str, pd.DataFrame]:
        """Analyze geographical distribution of crashes."""
        try:
            if self.crash_data is None or self.crash_data.empty:
                logger.warning("No data available for geographical analysis")
                return {}
            
            # Count crashes by location
            location_counts = self.crash_data['location'].value_counts().reset_index()
            location_counts.columns = ['location', 'count']
            
            # Get top locations
            top_locations = location_counts.nlargest(10, 'count')
            
            return {
                'location_counts': location_counts,
                'top_locations': top_locations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing geographical patterns: {e}", exc_info=True)
            return {}
    
    def _analyze_causal_factors(self) -> Dict[str, pd.DataFrame]:
        """Analyze causal factors in crash descriptions."""
        try:
            if self.crash_data is None or self.crash_data.empty:
                logger.warning("No data available for causal analysis")
                return {}
            
            # Extract factors from descriptions
            # For now, use a simple keyword-based approach
            keywords = {
                'weather': ['weather', 'rain', 'snow', 'wind', 'storm', 'fog'],
                'mechanical': ['engine', 'mechanical', 'failure', 'malfunction', 'gear'],
                'human': ['pilot', 'error', 'fatigue', 'mistake', 'oversight'],
                'environmental': ['bird', 'terrain', 'obstacle', 'runway'],
                'unknown': ['unknown', 'unclear', 'investigation']
            }
            
            # Count occurrences of each factor
            factor_counts = {factor: 0 for factor in keywords}
            
            for desc in self.crash_data['description']:
                desc = str(desc).lower()
                for factor, terms in keywords.items():
                    if any(term in desc for term in terms):
                        factor_counts[factor] += 1
            
            # Convert to DataFrame
            factors_df = pd.DataFrame([
                {'factor': k, 'count': v}
                for k, v in factor_counts.items()
            ]).sort_values('count', ascending=False)
            
            return {
                'factor_counts': factors_df,
                'total_analyzed': len(self.crash_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing causal factors: {e}", exc_info=True)
            return {}
    
    def _analyze_severity(self) -> Dict[str, float]:
        """Analyze crash severity patterns."""
        try:
            if 'fatalities' not in self.crash_data.columns:
                return {}
            
            severity_stats = {
                'mean_fatalities': float(self.crash_data['fatalities'].mean()),
                'max_fatalities': float(self.crash_data['fatalities'].max()),
                'total_fatalities': float(self.crash_data['fatalities'].sum()),
                'severity_distribution': self.crash_data['fatalities'].value_counts()
            }
            
            return severity_stats
            
        except Exception as e:
            logger.error(f"Error in severity analysis: {e}")
            return {}
    
    def _analyze_text_data(self) -> Dict[str, List[Tuple[str, int]]]:
        """Analyze textual crash descriptions."""
        try:
            if 'description' not in self.crash_data.columns:
                return {}
            
            # Combine all text for analysis
            all_text = ' '.join(self.crash_data['description'].fillna(''))
            doc = self.nlp(all_text)
            
            # Extract entities and key terms
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            key_terms = [token.text for token in doc 
                        if not token.is_stop and token.is_alpha]
            
            return {
                'named_entities': Counter(entities).most_common(10),
                'key_terms': Counter(key_terms).most_common(10)
            }
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {}
    
    def generate_visualizations(self, output_dir: str = "reports/figures") -> None:
        """Generate visualizations of analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Temporal trends
            if 'date' in self.crash_data.columns:
                temporal_fig = self._plot_temporal_trends()
                temporal_fig.write_html(output_dir / 'temporal_trends.html')
            
            # Geographical distribution
            if 'location' in self.crash_data.columns:
                geo_fig = self._plot_geographical_distribution()
                geo_fig.write_html(output_dir / 'geographical_distribution.html')
            
            # Severity distribution
            if 'fatalities' in self.crash_data.columns:
                severity_fig = self._plot_severity_distribution()
                severity_fig.write_html(output_dir / 'severity_distribution.html')
            
            logger.info(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _plot_temporal_trends(self) -> go.Figure:
        """Plot temporal trends in crash data."""
        yearly_counts = self.crash_data.groupby(
            self.crash_data['date'].dt.year
        ).size()
        
        fig = px.line(
            x=yearly_counts.index,
            y=yearly_counts.values,
            title='Yearly Crash Trends',
            labels={'x': 'Year', 'y': 'Number of Crashes'}
        )
        return fig
    
    def _plot_geographical_distribution(self) -> go.Figure:
        """Plot geographical distribution of crashes."""
        location_counts = self.crash_data['location'].value_counts().head(10)
        
        fig = px.bar(
            x=location_counts.index,
            y=location_counts.values,
            title='Top 10 Crash Locations',
            labels={'x': 'Location', 'y': 'Number of Crashes'}
        )
        return fig
    
    def _plot_severity_distribution(self) -> go.Figure:
        """Plot distribution of crash severity."""
        fig = px.histogram(
            self.crash_data,
            x='fatalities',
            title='Distribution of Fatalities',
            labels={'fatalities': 'Number of Fatalities', 'count': 'Frequency'}
        )
        return fig

# Example usage
def main():
    # Initialize analyzer
    analyzer = DataAnalyzer()
    
    # Analyze data
    patterns = analyzer.analyze_patterns()
    print("Analysis Results:")
    for key, value in patterns.items():
        print(f"\n{key.upper()}:")
        print(value)
        
    # Generate visualizations
    analyzer.generate_visualizations()

if __name__ == "__main__":
    main()
