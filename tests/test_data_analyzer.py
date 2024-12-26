import pytest
import pandas as pd
from pathlib import Path
from src.analysis.data_analyzer import DataAnalyzer

@pytest.fixture
def analyzer():
    return DataAnalyzer(data_dir="tests/test_data")

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
        'location': ['Location A', 'Location B', 'Location C'],
        'description': ['Crash A', 'Crash B', 'Crash C'],
        'fatalities': [0, 2, 1]
    })

def test_load_data(analyzer, tmp_path, sample_data):
    # Save sample data
    test_file = tmp_path / "crash_data_test.csv"
    sample_data.to_csv(test_file, index=False)
    
    # Test loading data
    loaded_data = analyzer.load_data(str(test_file))
    assert isinstance(loaded_data, pd.DataFrame)
    assert len(loaded_data) == len(sample_data)

def test_analyze_patterns(analyzer, sample_data):
    analyzer.crash_data = sample_data
    patterns = analyzer.analyze_patterns()
    
    assert isinstance(patterns, dict)
    assert 'temporal_patterns' in patterns
    assert 'geographical_patterns' in patterns
    assert 'causal_factors' in patterns
    assert 'severity_analysis' in patterns
    assert 'text_insights' in patterns

def test_temporal_patterns(analyzer, sample_data):
    analyzer.crash_data = sample_data
    patterns = analyzer._analyze_temporal_patterns()
    
    assert isinstance(patterns, dict)
    assert 'yearly' in patterns
    assert 'monthly' in patterns
    assert 'daily' in patterns

def test_geographical_patterns(analyzer, sample_data):
    analyzer.crash_data = sample_data
    patterns = analyzer._analyze_geographical_patterns()
    
    assert isinstance(patterns, dict)
    assert 'location_frequency' in patterns
    assert 'top_locations' in patterns
    assert len(patterns['top_locations']) == len(sample_data)

def test_causal_factors(analyzer, sample_data):
    analyzer.crash_data = sample_data
    factors = analyzer._analyze_causal_factors()
    
    assert isinstance(factors, dict)
    assert 'common_phrases' in factors
    assert 'causal_factors' in factors

def test_severity_analysis(analyzer, sample_data):
    analyzer.crash_data = sample_data
    severity = analyzer._analyze_severity()
    
    assert isinstance(severity, dict)
    assert 'mean_fatalities' in severity
    assert 'max_fatalities' in severity
    assert 'total_fatalities' in severity
    assert severity['mean_fatalities'] == 1.0
    assert severity['max_fatalities'] == 2.0
    assert severity['total_fatalities'] == 3.0

def test_text_analysis(analyzer, sample_data):
    analyzer.crash_data = sample_data
    insights = analyzer._analyze_text_data()
    
    assert isinstance(insights, dict)
    assert 'named_entities' in insights
    assert 'key_terms' in insights

def test_generate_visualizations(analyzer, sample_data, tmp_path):
    analyzer.crash_data = sample_data
    output_dir = tmp_path / "figures"
    analyzer.generate_visualizations(str(output_dir))
    
    assert output_dir.exists()
    assert (output_dir / "temporal_trends.html").exists()
    assert (output_dir / "geographical_distribution.html").exists()
    assert (output_dir / "severity_distribution.html").exists()
