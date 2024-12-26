import pytest
import asyncio
import pandas as pd
from pathlib import Path
from src.collectors.crash_data_collector import CrashDataCollector

@pytest.fixture
def collector():
    return CrashDataCollector(data_dir="tests/test_data")

@pytest.fixture
def sample_html():
    return """
    <div class="crash-item">
        <span class="date">2023-01-01</span>
        <span class="location">New York, USA</span>
        <span class="aircraft">Boeing 737</span>
        <span class="fatalities">0</span>
        <p class="description">Emergency landing due to technical issues</p>
    </div>
    """

@pytest.mark.asyncio
async def test_fetch_page(collector):
    async with collector:
        # Test with a known reliable URL
        html = await collector.fetch_page("https://example.com")
        assert isinstance(html, str)
        assert len(html) > 0

@pytest.mark.asyncio
async def test_parse_crash_data(collector, sample_html):
    crashes = await collector.parse_crash_data(sample_html)
    assert len(crashes) == 1
    assert crashes[0]['date'] == '2023-01-01'
    assert crashes[0]['location'] == 'New York, USA'
    assert crashes[0]['aircraft_type'] == 'Boeing 737'
    assert crashes[0]['fatalities'] == '0'

@pytest.mark.asyncio
async def test_collect_data(collector):
    # Test with mock URLs
    urls = ["https://example.com/page1", "https://example.com/page2"]
    df = await collector.collect_data(urls)
    assert isinstance(df, pd.DataFrame)

def test_load_latest_data(collector, tmp_path):
    # Create a test CSV file in the collector's data directory
    test_data = pd.DataFrame({
        'date': ['2023-01-01'],
        'location': ['Test Location'],
        'description': ['Test Description']
    })
    collector.data_dir.mkdir(parents=True, exist_ok=True)
    test_file = collector.data_dir / "crash_data_20231225.csv"
    test_data.to_csv(test_file, index=False)

    # Test loading the data
    loaded_data = collector.load_latest_data()
    assert isinstance(loaded_data, pd.DataFrame)
    assert not loaded_data.empty
    assert 'date' in loaded_data.columns
