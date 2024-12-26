import os
import sys
from pathlib import Path
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import logging
import dash_leaflet as dl
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from plotly.subplots import make_subplots
import calendar
import shap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
import base64
import xlsxwriter
import dash_table
import dash_bootstrap_components as dbc
from flask_caching import Cache
from dash import DiskcacheManager
import diskcache

# Add parent directory to path for imports
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.append(root_dir)

from src.config.data_sources import get_data_sources
from src.collectors.crash_data_collector import CrashDataCollector
from src.analysis.data_analyzer import DataAnalyzer
from src.models.ai_model import AIModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
logger.info(f"Root directory: {root_dir}")
data_dir = os.path.join(root_dir, "data", "raw")
models_dir = os.path.join(root_dir, "models")

logger.info(f"Data directory: {data_dir}")
logger.info(f"Models directory: {models_dir}")

# Initialize collector and analyzer with MongoDB
collector = CrashDataCollector(data_dir=data_dir)
analyzer = DataAnalyzer(data_dir=data_dir)

# Get available sources
available_sources = collector.get_available_sources()
logger.info(f"Found {len(available_sources)} data sources")

# Initialize model
try:
    model = AIModel(model_dir=models_dir)
    model.load_models()
    logger.info("Successfully loaded models")
except Exception as e:
    logger.warning(f"Could not load model: {e}")
    model = None

# Initialize geocoder with increased timeout
geocoder = Nominatim(user_agent="aviation_crash_analysis", timeout=5)

def get_coordinates(location):
    """Get coordinates for a location string."""
    try:
        # Add caching to avoid repeated requests
        location_data = geocoder.geocode(location, exactly_one=True)
        if location_data:
            return [location_data.latitude, location_data.longitude]
    except GeocoderTimedOut:
        logger.warning(f"Geocoding timed out for location: {location}")
    except Exception as e:
        logger.error(f"Error geocoding location {location}: {e}")
    return None

# Suppress callback exceptions for dynamic components
cache_dir = ".cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

background_callback_manager = DiskcacheManager(
    diskcache.Cache(cache_dir)
)

cache_config = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': cache_dir,
    'CACHE_DEFAULT_TIMEOUT': 300
}

app = dash.Dash(__name__, 
                suppress_callback_exceptions=True, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                background_callback_manager=background_callback_manager)

app.title = "Aviation Crash Analysis Dashboard"

cache = Cache()
cache.init_app(app.server, config=cache_config)

# Custom CSS for the loading spinner
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-color: #2c3e50;
                --secondary-color: #34495e;
                --accent-color: #3498db;
                --success-color: #2ecc71;
                --warning-color: #f1c40f;
                --error-color: #e74c3c;
                --background-color: #f5f7fa;
                --card-background: #ffffff;
                --text-color: #2c3e50;
                --border-radius: 8px;
                --spacing-unit: 16px;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0;
                background-color: var(--background-color);
                color: var(--text-color);
                line-height: 1.6;
            }
            
            .dashboard-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: var(--spacing-unit);
            }
            
            .dashboard-title {
                color: var(--primary-color);
                text-align: center;
                margin: calc(var(--spacing-unit) * 2) 0;
                font-size: 2.5em;
                font-weight: 300;
                border-bottom: 2px solid var(--accent-color);
                padding-bottom: var(--spacing-unit);
            }
            
            .control-panel {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: var(--spacing-unit);
                margin-bottom: calc(var(--spacing-unit) * 2);
                background: var(--card-background);
                padding: var(--spacing-unit);
                border-radius: var(--border-radius);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .source-controls {
                display: flex;
                flex-direction: column;
                gap: var(--spacing-unit);
            }
            
            .source-selector, .date-range-selector {
                padding: calc(var(--spacing-unit) / 2);
                background: var(--background-color);
                border-radius: calc(var(--border-radius) / 2);
            }
            
            .source-selector label, .date-range-selector label {
                display: block;
                margin-bottom: calc(var(--spacing-unit) / 2);
                font-weight: 500;
                color: var(--secondary-color);
            }
            
            .refresh-button {
                background-color: var(--success-color);
                color: white;
                padding: calc(var(--spacing-unit) / 2) var(--spacing-unit);
                border: none;
                border-radius: calc(var(--border-radius) / 2);
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.3s ease;
            }
            
            .refresh-button:hover {
                background-color: #27ae60;
            }
            
            .stats-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            
            .stats-card {
                background: white;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            
            .stats-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
            
            .stats-card h4 {
                margin: 0 0 0.5rem 0;
                color: var(--accent-color);
                font-size: 1.1rem;
            }
            
            .stats-card ul {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            
            .stats-card li {
                margin: 0.5rem 0;
                font-size: 0.9rem;
                color: #666;
            }
            
            .stats-card li strong {
                color: #333;
            }
            
            .tabs-container {
                margin-top: calc(var(--spacing-unit) * 2);
            }
            
            .tab-content {
                background: var(--card-background);
                padding: var(--spacing-unit);
                border-radius: var(--border-radius);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .analysis-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: calc(var(--spacing-unit) * 2);
                padding: var(--spacing-unit);
            }
            
            .map-container {
                grid-column: 1 / -1;
                background: var(--card-background);
                padding: var(--spacing-unit);
                border-radius: var(--border-radius);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .graph-container {
                background: var(--card-background);
                padding: var(--spacing-unit);
                border-radius: var(--border-radius);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .error-text {
                color: var(--error-color);
                padding: var(--spacing-unit);
                border-radius: var(--border-radius);
                background-color: #fde8e8;
                border: 1px solid #fbd5d5;
            }
            
            .warning-text {
                color: var(--warning-color);
                padding: var(--spacing-unit);
                border-radius: var(--border-radius);
                background-color: #fef6e7;
                border: 1px solid #fdecd0;
            }

            /* Custom styles for Dash components */
            .Select-control {
                border-color: #e2e8f0 !important;
                border-radius: calc(var(--border-radius) / 2) !important;
            }
            
            .DateRangePickerInput {
                border-color: #e2e8f0 !important;
                border-radius: calc(var(--border-radius) / 2) !important;
            }
            
            .DateInput_input {
                font-size: 14px !important;
                padding: 8px !important;
            }
            
            .incident-textarea {
                width: 100%;
                padding: var(--spacing-unit);
                border: 1px solid #e2e8f0;
                border-radius: var(--border-radius);
                font-size: 14px;
                resize: vertical;
                margin-bottom: var(--spacing-unit);
            }
            
            .analyze-button {
                background-color: var(--accent-color);
                color: white;
                padding: calc(var(--spacing-unit) / 2) var(--spacing-unit);
                border: none;
                border-radius: calc(var(--border-radius) / 2);
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.3s ease;
                margin-bottom: var(--spacing-unit);
            }
            
            .analyze-button:hover {
                background-color: #2980b9;
            }
            
            .risk-output, .prediction-results {
                margin-top: var(--spacing-unit);
                padding: var(--spacing-unit);
                background: var(--background-color);
                border-radius: var(--border-radius);
            }
            
            .model-info {
                margin-bottom: var(--spacing-unit);
                padding: var(--spacing-unit);
                background: var(--background-color);
                border-radius: var(--border-radius);
            }
            
            .prediction-content, .model-content {
                padding: var(--spacing-unit);
                background: var(--card-background);
                border-radius: var(--border-radius);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Loading spinner */
            ._dash-loading {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 9999;
                background: rgba(255, 255, 255, 0.8);
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: none;
            }
            
            ._dash-loading::after {
                content: " ";
                display: block;
                width: 40px;
                height: 40px;
                margin: 8px auto;
                border-radius: 50%;
                border: 6px solid var(--accent-color);
                border-color: var(--accent-color) transparent var(--accent-color) transparent;
                animation: loading-spinner 1.2s linear infinite;
            }
            
            @keyframes loading-spinner {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            ._dash-loading.loading { display: block; }
            
            /* Risk Prediction Styles */
            .risk-prediction-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
                padding: 1rem;
            }
            
            .risk-input-section {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .additional-inputs {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1rem 0;
            }
            
            .input-field {
                margin-bottom: 1rem;
            }
            
            .risk-output-section {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .risk-level {
                font-size: 1.5rem;
                font-weight: bold;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                text-align: center;
                margin: 1rem 0;
            }
            
            .risk-level.high {
                background: #ffebee;
                color: #c62828;
            }
            
            .risk-level.medium {
                background: #fff3e0;
                color: #ef6c00;
            }
            
            .risk-level.low {
                background: #e8f5e9;
                color: #2e7d32;
            }
            
            .confidence-level {
                font-size: 1.2rem;
                color: #666;
                text-align: center;
            }
            
            .similar-cases-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 1rem;
            }
            
            .similar-cases-table th,
            .similar-cases-table td {
                padding: 0.5rem;
                text-align: left;
                border-bottom: 1px solid #eee;
            }
            
            .similar-cases-table th {
                background: #f5f5f5;
                font-weight: 500;
            }
            
            .analyze-button {
                width: 100%;
                margin-top: 1rem;
            }
            
            /* Filter Panel Styles */
            .filters-section {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 1.5rem;
            }
            
            .filter-panel {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            
            .filter-item {
                margin-bottom: 1rem;
            }
            
            .search-box {
                width: 100%;
                padding: 0.5rem;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            
            .export-container {
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            
            .download-link {
                color: var(--accent-color);
                text-decoration: none;
                padding: 0.5rem 1rem;
                border: 1px solid var(--accent-color);
                border-radius: 4px;
                transition: all 0.2s;
            }
            
            .download-link:hover {
                background: var(--accent-color);
                color: white;
            }
            
            /* Table Styles */
            .table-container {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Modal Styles */
            .incident-description {
                margin-top: 1rem;
                padding: 1rem;
                background: #f5f5f5;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div id="react-entry-point">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = html.Div([
    html.H1("Aviation Crash Analysis Dashboard", className="dashboard-title"),
    
    # Control Panel
    html.Div([
        # Left side - Controls
        html.Div([
            # Source Selection
            html.Div([
                html.Label("Select Sources:"),
                dcc.Dropdown(
                    id="data-sources",
                    options=[
                        {
                            'label': f"{source['_id']} ({source['count']} records)",
                            'value': source['_id']
                        }
                        for source in available_sources
                    ],
                    value=[source['_id'] for source in available_sources],
                    multi=True,
                    className="source-dropdown"
                ),
            ], className="source-selector"),
            
            # Date Range Selection
            html.Div([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=datetime(2000, 1, 1),
                    max_date_allowed=datetime.now(),
                    initial_visible_month=datetime.now(),
                    end_date=datetime.now(),
                    start_date=datetime(2020, 1, 1),
                    className="date-picker"
                ),
            ], className="date-range-selector"),
            
            # Source Statistics as Cards
            html.Div([
                html.Div(id="source-stats", className="stats-cards")
            ], className="stats-section"),
            
            # Refresh Button
            html.Button(
                "Refresh Data",
                id="refresh-data",
                className="refresh-button"
            ),
        ], className="source-controls"),
        
        # Right side - Stats
        html.Div([
            html.H4("Data Status"),
            html.Div(id="data-status", className="data-status"),
        ], className="source-stats"),
    ], className="control-panel"),
    
    # Tabs
    dcc.Tabs([
        # Analysis Tab
        dcc.Tab(label="Analysis", children=[
            html.Div([
                # Trend Analysis Section
                html.Div([
                    html.H3("Trend Analysis"),
                    dcc.Loading(
                        id="trend-loading",
                        type="default",
                        children=[dcc.Graph(id="trend-analysis")]
                    )
                ], className="graph-container"),
                
                # Aircraft Analysis Section
                html.Div([
                    html.H3("Aircraft Analysis"),
                    dcc.Loading(
                        id="aircraft-analysis-loading",
                        type="default",
                        children=[dcc.Graph(id="aircraft-analysis")]
                    )
                ], className="graph-container"),
                
                # Map View
                html.Div([
                    html.H3("Crash Locations"),
                    dcc.Loading(
                        id="map-loading",
                        type="default",
                        children=[
                            dl.Map([
                                dl.TileLayer(),
                                dl.LayerGroup(id="crash-locations")
                            ], style={'width': '100%', 'height': '500px'}, id="crash-map")
                        ]
                    )
                ], className="map-container"),
                
                # Analysis Graphs
                html.Div([
                    html.H3("Temporal Analysis"),
                    dcc.Loading(
                        id="temporal-loading",
                        type="default",
                        children=[dcc.Graph(id="temporal-graph")]
                    )
                ], className="graph-container"),
                
                html.Div([
                    html.H3("Source Comparison"),
                    dcc.Loading(
                        id="comparison-loading",
                        type="default",
                        children=[dcc.Graph(id="source-comparison")]
                    )
                ], className="graph-container"),
                
                html.Div([
                    html.H3("Severity Distribution"),
                    dcc.Loading(
                        id="severity-loading",
                        type="default",
                        children=[dcc.Graph(id="severity-distribution")]
                    )
                ], className="graph-container"),
                
                html.Div([
                    html.H3("Aircraft Types"),
                    dcc.Loading(
                        id="aircraft-loading",
                        type="default",
                        children=[dcc.Graph(id="aircraft-types")]
                    )
                ], className="graph-container")
            ], className="analysis-grid")
        ]),
        
        # Risk Prediction Tab
        dcc.Tab(label="Risk Prediction", children=[
            html.Div([
                # Input Section
                html.Div([
                    html.H3("Risk Assessment Input"),
                    dcc.Textarea(
                        id='incident-description',
                        placeholder='Enter incident description...',
                        style={'width': '100%', 'height': 100},
                        className="incident-textarea"
                    ),
                    
                    # Additional Input Fields
                    html.Div([
                        html.Div([
                            html.Label("Aircraft Type:"),
                            dcc.Dropdown(
                                id='aircraft-type-input',
                                options=[
                                    {'label': 'Commercial Jet', 'value': 'commercial_jet'},
                                    {'label': 'Private Aircraft', 'value': 'private_aircraft'},
                                    {'label': 'Helicopter', 'value': 'helicopter'},
                                    {'label': 'Military', 'value': 'military'}
                                ],
                                className="input-dropdown"
                            )
                        ], className="input-field"),
                        
                        html.Div([
                            html.Label("Weather Conditions:"),
                            dcc.Dropdown(
                                id='weather-input',
                                options=[
                                    {'label': 'Clear', 'value': 'clear'},
                                    {'label': 'Rain', 'value': 'rain'},
                                    {'label': 'Snow', 'value': 'snow'},
                                    {'label': 'Fog', 'value': 'fog'}
                                ],
                                className="input-dropdown"
                            )
                        ], className="input-field"),
                        
                        html.Div([
                            html.Label("Flight Phase:"),
                            dcc.Dropdown(
                                id='flight-phase-input',
                                options=[
                                    {'label': 'Takeoff', 'value': 'takeoff'},
                                    {'label': 'Landing', 'value': 'landing'},
                                    {'label': 'Cruise', 'value': 'cruise'},
                                    {'label': 'Approach', 'value': 'approach'}
                                ],
                                className="input-dropdown"
                            )
                        ], className="input-field")
                    ], className="additional-inputs"),
                    
                    html.Button(
                        'Analyze Risk',
                        id='analyze-button',
                        className="analyze-button"
                    )
                ], className="risk-input-section"),
                
                # Results Section
                html.Div([
                    dcc.Loading(
                        id="risk-loading",
                        type="default",
                        children=[
                            # Risk Score
                            html.Div([
                                html.H4("Risk Assessment Results"),
                                html.Div(id='risk-score', className="risk-score"),
                                html.Div(id='confidence-score', className="confidence-score")
                            ], className="risk-results"),
                            
                            # Feature Importance
                            html.Div([
                                html.H4("Feature Importance"),
                                dcc.Graph(id='feature-importance-plot')
                            ], className="feature-importance"),
                            
                            # Similar Cases
                            html.Div([
                                html.H4("Similar Historical Cases"),
                                html.Div(id='similar-cases', className="similar-cases")
                            ], className="similar-cases-section")
                        ]
                    )
                ], className="risk-output-section")
            ], className="risk-prediction-content")
        ]),
        
        # Model Information Tab
        dcc.Tab(label="Model Information", children=[
            html.Div([
                html.H3("Model Details"),
                dcc.Loading(
                    id="model-loading",
                    type="default",
                    children=[
                        html.Div(id="model-info", className="model-info"),
                        dcc.Graph(id="training-history")
                    ]
                ),
                dcc.Interval(
                    id='model-info-interval',
                    interval=30000,  # 30 seconds
                    n_intervals=0
                )
            ], className="model-content")
        ]),
        
        # Data Exploration Tab
        dcc.Tab(label="Data Exploration", children=[
            html.Div([
                # Advanced Filter Panel
                html.Div([
                    html.Div([
                        html.H3("Advanced Filters"),
                        html.Button(
                            "Toggle Filters",
                            id="toggle-filters",
                            className="toggle-button"
                        ),
                        dcc.Loading(
                            id="filter-loading",
                            type="default",
                            children=[
                                html.Div([
                                    # Severity Filter
                                    html.Div([
                                        html.Label("Severity:"),
                                        dcc.Dropdown(
                                            id='severity-filter',
                                            options=[
                                                {'label': 'Fatal', 'value': 'Fatal'},
                                                {'label': 'Serious', 'value': 'Serious'},
                                                {'label': 'Minor', 'value': 'Minor'}
                                            ],
                                            multi=True,
                                            className="filter-dropdown"
                                        )
                                    ], className="filter-item"),
                                    
                                    # Aircraft Type Filter
                                    html.Div([
                                        html.Label("Aircraft Type:"),
                                        dcc.Dropdown(
                                            id='aircraft-filter',
                                            options=[],  # Will be populated from data
                                            multi=True,
                                            className="filter-dropdown"
                                        )
                                    ], className="filter-item"),
                                    
                                    # Operator Filter
                                    html.Div([
                                        html.Label("Operator:"),
                                        dcc.Dropdown(
                                            id='operator-filter',
                                            options=[],  # Will be populated from data
                                            multi=True,
                                            className="filter-dropdown"
                                        )
                                    ], className="filter-item"),
                                    
                                    # Search Box
                                    html.Div([
                                        html.Label("Search:"),
                                        dcc.Input(
                                            id='search-input',
                                            type='text',
                                            placeholder='Search in incident descriptions...',
                                            className="search-box"
                                        )
                                    ], className="filter-item"),
                                    
                                    # Export Button
                                    html.Div([
                                        html.Button(
                                            "Export Data",
                                            id="export-button",
                                            className="export-button"
                                        ),
                                        html.Div(id="download-link-container")
                                    ], className="export-container")
                                ], id="filter-panel", className="filter-panel")
                            ]
                        )
                    ], className="filters-section"),
                    
                    # Data Table with Details
                    html.Div([
                        html.H3("Incident List"),
                        dcc.Loading(
                            id="table-loading",
                            type="default",
                            children=[
                                dash_table.DataTable(
                                    id='incident-table',
                                    columns=[
                                        {"name": "Date", "id": "date"},
                                        {"name": "Location", "id": "location"},
                                        {"name": "Severity", "id": "severity"},
                                        {"name": "Aircraft", "id": "aircraft_type"},
                                        {"name": "Operator", "id": "operator"}
                                    ],
                                    style_data={
                                        'whiteSpace': 'normal',
                                        'height': 'auto',
                                    },
                                    style_cell={
                                        'textAlign': 'left',
                                        'padding': '10px'
                                    },
                                    style_header={
                                        'backgroundColor': 'rgb(230, 230, 230)',
                                        'fontWeight': 'bold'
                                    },
                                    page_size=10,
                                    filter_action="native",
                                    sort_action="native",
                                    sort_mode="multi",
                                    row_selectable="single",
                                    selected_rows=[],
                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(248, 248, 248)'
                                        }
                                    ]
                                )
                            ]
                        )
                    ], className="table-container"),
                    
                    # Incident Details Modal
                    dbc.Modal([
                        dbc.ModalHeader("Incident Details"),
                        dbc.ModalBody(id="incident-details-body"),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close-modal", className="ml-auto")
                        )
                    ], id="incident-details-modal", size="lg")
                ], className="data-exploration-section")
            ], className="tab-content")
        ])
    ], className="tabs-container")
], className="dashboard-container")

# Cache data loading functions
@cache.memoize(timeout=300)
def load_source_data():
    """Load and cache data sources."""
    return collector.get_available_sources()

@cache.memoize(timeout=60)
def load_crash_data(sources, filters=None):
    """Load and cache crash data."""
    return collector.get_data(sources, filters=filters)

@cache.memoize(timeout=300)
def load_statistics(sources):
    """Load and cache statistics."""
    return collector.get_statistics(sources)

# Callbacks
@app.callback(
    Output("temporal-graph", "figure"),
    [Input("data-sources", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_temporal_analysis(sources, start_date, end_date):
    """Update temporal analysis visualization."""
    if not sources:
        return empty_figure("No sources selected")
    
    try:
        # Convert dates
        start = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d') if start_date else None
        end = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d') if end_date else None
        
        # Collect temporal data from all sources
        all_data = pd.DataFrame(columns=['date', 'source'])
        
        for source in sources:
            df = collector.get_source_data(source, start, end)
            if not df.empty and 'date' in df.columns:
                source_data = pd.DataFrame({
                    'date': pd.to_datetime(df['date']),
                    'source': source
                })
                all_data = pd.concat([all_data, source_data], ignore_index=True)
        
        if len(all_data) == 0:
            return empty_figure("No temporal data available")
        
        # Create monthly counts
        all_data['month'] = all_data['date'].dt.to_period('M')
        monthly_counts = all_data.groupby(['month', 'source']).size().unstack(fill_value=0)
        monthly_counts.index = monthly_counts.index.astype(str)
        
        # Create line plot
        fig = go.Figure()
        
        # Add trace for each source
        for source in sources:
            if source in monthly_counts.columns:
                fig.add_trace(go.Scatter(
                    x=monthly_counts.index,
                    y=monthly_counts[source],
                    name=source,
                    mode='lines+markers',
                    hovertemplate="<b>%{x}</b><br>" +
                                "Source: " + source + "<br>" +
                                "Incidents: %{y}<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            title='Monthly Incident Trends by Source',
            xaxis_title='Month',
            yaxis_title='Number of Incidents',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating temporal analysis: {e}", exc_info=True)
        return empty_figure(f"Error: {str(e)}")

@app.callback(
    Output("source-comparison", "figure"),
    [Input("data-sources", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_source_comparison(sources, start_date, end_date):
    """Update source comparison visualization."""
    if not sources:
        return empty_figure("No sources selected")
    
    try:
        # Convert dates
        start = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d') if start_date else None
        end = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d') if end_date else None
        
        # Collect data for each source
        source_data = []
        for source in sources:
            df = collector.get_source_data(source, start, end)
            if not df.empty:
                # Safely check for severity
                fatal_count = 0
                if 'severity' in df.columns:
                    fatal_count = len(df[df['severity'].str.contains('Fatal', case=False, na=False)])
                
                # Get aircraft types with explicit dtype
                aircraft_types = pd.Series(dtype='object')
                if 'aircraft_type' in df.columns:
                    aircraft_types = df['aircraft_type'].value_counts()
                
                source_data.append({
                    'source': source,
                    'total': len(df),
                    'fatal': fatal_count,
                    'aircraft_types': aircraft_types.to_dict()
                })
        
        if not source_data:
            return empty_figure("No data available")
            
        # Create comparison figure
        fig = go.Figure()
        
        # Add total incidents bar
        fig.add_trace(go.Bar(
            name='Total Incidents',
            x=[d['source'] for d in source_data],
            y=[d['total'] for d in source_data],
            marker_color='#1f77b4'
        ))
        
        # Add fatal incidents bar
        fig.add_trace(go.Bar(
            name='Fatal Incidents',
            x=[d['source'] for d in source_data],
            y=[d['fatal'] for d in source_data],
            marker_color='#d62728'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Incident Comparison by Source',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            barmode='group',
            xaxis_title='Data Source',
            yaxis_title='Number of Incidents',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            bargap=0.15,
            bargroupgap=0.1,
            height=400,
            margin=dict(t=100, b=50, l=50, r=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        # Update axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating source comparison: {e}", exc_info=True)
        return empty_figure(f"Error: {str(e)}")

@app.callback(
    Output("aircraft-types", "figure"),
    [Input("data-sources", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_aircraft_types(sources, start_date, end_date):
    """Update aircraft types visualization."""
    if not sources:
        return empty_figure("No sources selected")
    
    try:
        # Convert dates
        start = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d') if start_date else None
        end = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d') if end_date else None
        
        # Create empty DataFrame with proper columns
        all_data = pd.DataFrame(columns=['aircraft_type', 'source'])
        
        # Collect aircraft type data from all sources
        for source in sources:
            df = collector.get_source_data(source, start, end)
            if not df.empty and 'aircraft_type' in df.columns:
                source_data = pd.DataFrame({
                    'aircraft_type': df['aircraft_type'],
                    'source': source
                })
                all_data = pd.concat([all_data, source_data], ignore_index=True)
        
        if len(all_data) == 0:
            return empty_figure("No aircraft type data available for selected sources")
        
        # Get top 10 aircraft types
        type_counts = all_data.groupby(['aircraft_type', 'source']).size().unstack(fill_value=0)
        top_types = type_counts.sum(axis=1).nlargest(10).index
        type_counts = type_counts.loc[top_types]
        
        # Create grouped bar chart
        fig = go.Figure()
        
        # Add trace for each source
        for source in sources:
            if source in type_counts.columns:
                fig.add_trace(go.Bar(
                    name=source,
                    x=type_counts.index,
                    y=type_counts[source],
                    hovertemplate="<b>%{x}</b><br>" +
                                "Source: " + source + "<br>" +
                                "Count: %{y}<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Top 10 Aircraft Types by Source',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            barmode='group',
            xaxis_title='Aircraft Type',
            yaxis_title='Number of Incidents',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,  # Increased height for better readability
            margin=dict(t=100, b=150, l=50, r=50),  # Increased bottom margin for labels
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            xaxis=dict(
                tickangle=45,
                tickmode='array',
                ticktext=[text[:20] + '...' if len(text) > 20 else text for text in type_counts.index],
                tickvals=list(range(len(type_counts.index)))
            )
        )
        
        # Update axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating aircraft types: {e}", exc_info=True)
        return empty_figure(f"Error: {str(e)}")

@app.callback(
    Output("prediction-results", "children"),
    [Input("analyze-button", "n_clicks")],
    [State("incident-description", "value")]
)
def make_prediction(n_clicks, description):
    """Make prediction based on incident description."""
    if not description:
        return html.Div("Please enter an incident description.", className="warning-text")
        
    try:
        # Get prediction from model
        prediction = model.predict(description)
        
        # Format prediction results
        return html.Div([
            html.H4("Risk Assessment Results"),
            
            # Risk Level
            html.Div([
                html.Label("Risk Level:", className="result-label"),
                html.Span(
                    f"{prediction['risk_level']}",
                    style={
                        "color": {
                            "High": "#dc3545",
                            "Medium": "#ffc107",
                            "Low": "#28a745"
                        }.get(prediction['risk_level'], "#6c757d"),
                        "fontWeight": "bold",
                        "marginLeft": "10px"
                    }
                )
            ], className="result-item"),
            
            # Confidence Score
            html.Div([
                html.Label("Confidence:", className="result-label"),
                html.Span(
                    f"{prediction['confidence']:.1%}",
                    style={"marginLeft": "10px"}
                )
            ], className="result-item"),
            
            # Contributing Factors
            html.Div([
                html.Label("Key Factors:", className="result-label"),
                html.Ul([
                    html.Li(factor) for factor in prediction['factors']
                ], style={"marginTop": "5px"})
            ], className="result-item"),
            
            # Recommendations
            html.Div([
                html.Label("Recommendations:", className="result-label"),
                html.P(prediction['recommendations'], 
                      style={"marginTop": "5px"})
            ], className="result-item")
        ], className="prediction-results")
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}", exc_info=True)
        return html.Div([
            html.P("Error making prediction:", style={"color": "#dc3545"}),
            html.P(str(e))
        ], className="error-text")

@app.callback(
    [Output("model-info", "children"),
     Output("training-history", "figure")],
    Input("model-info-interval", "n_intervals")
)
def update_model_info(n):
    """Update model information and training history."""
    try:
        # Placeholder for model info
        model_info = html.Div([
            html.H4("Model Parameters"),
            html.P("Architecture: Neural Network"),
            html.P("Input Features: 10"),
            html.P("Hidden Layers: 3"),
            html.P("Accuracy: 85%")
        ])
        
        # Placeholder for training history
        history_fig = go.Figure()
        history_fig.add_trace(go.Scatter(
            y=[0.6, 0.7, 0.8, 0.85, 0.87],
            name='Training Accuracy'
        ))
        history_fig.add_trace(go.Scatter(
            y=[0.55, 0.65, 0.75, 0.8, 0.82],
            name='Validation Accuracy'
        ))
        history_fig.update_layout(
            title="Model Training History",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            template="plotly_white"
        )
        
        return model_info, history_fig
        
    except Exception as e:
        logger.error(f"Error updating model info: {e}", exc_info=True)
        return html.Div("Error loading model information"), go.Figure()

@app.callback(
    [
        Output("source-stats", "children"),
        Output("data-status", "children")
    ],
    [
        Input("data-sources", "value"),
        Input("refresh-data", "n_clicks")
    ],
    [
        State("date-range", "start_date"),
        State("date-range", "end_date")
    ]
)
def update_source_info(selected_sources, n_clicks, start_date, end_date):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    
    # Initialize outputs
    stats_cards = []
    status_message = None
    
    # Handle refresh button click
    if trigger_id == "refresh-data" and n_clicks:
        try:
            # Your existing refresh logic here
            status_message = html.Div("Data refreshed successfully!", style={"color": "green"})
        except Exception as e:
            status_message = html.Div(f"Error refreshing data: {str(e)}", style={"color": "red"})
    
    # Create stats cards
    if selected_sources:
        for source in available_sources:
            if source['_id'] in selected_sources:
                card = html.Div([
                    html.H4(source['_id']),
                    html.Ul([
                        html.Li([
                            html.Strong("Total Records: "), 
                            f"{source['count']}"
                        ]),
                        html.Li([
                            html.Strong("Date Range: "), 
                            f"{source.get('date_range', 'N/A')}"
                        ]),
                        html.Li([
                            html.Strong("Last Import: "), 
                            f"{source.get('last_import', 'Never')}"
                        ])
                    ])
                ], className="stats-card")
                stats_cards.append(card)
    
    # If no status message was set by refresh, use default
    if status_message is None:
        if selected_sources:
            status_message = html.Div(f"Showing data for {len(selected_sources)} source(s)")
        else:
            status_message = html.Div("No sources selected")
    
    return stats_cards, status_message

@app.callback(
    Output("severity-distribution", "figure"),
    [Input("data-sources", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_severity_distribution(sources, start_date, end_date):
    """Update severity distribution visualization."""
    if not sources:
        return empty_figure("No sources selected")
    
    try:
        # Convert dates
        start = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d') if start_date else None
        end = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d') if end_date else None
        
        # Collect severity data from all sources
        severity_data = {}
        for source in sources:
            df = collector.get_source_data(source, start, end)
            if not df.empty and 'severity' in df.columns:
                severity_counts = df['severity'].value_counts()
                for severity, count in severity_counts.items():
                    if severity not in severity_data:
                        severity_data[severity] = {}
                    severity_data[severity][source] = count
        
        if not severity_data:
            return empty_figure("No severity data available for selected sources")
            
        # Create stacked bar chart
        fig = go.Figure()
        
        # Add trace for each severity level
        for severity in severity_data:
            fig.add_trace(go.Bar(
                name=severity,
                x=sources,
                y=[severity_data[severity].get(source, 0) for source in sources],
                hovertemplate="<b>%{x}</b><br>" +
                            "Severity: " + severity + "<br>" +
                            "Count: %{y}<extra></extra>"
            ))
        
        # Update layout
        fig.update_layout(
            title='Severity Distribution by Source',
            barmode='stack',
            xaxis_title='Data Source',
            yaxis_title='Number of Incidents',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating severity distribution: {e}", exc_info=True)
        return empty_figure(f"Error: {str(e)}")

@app.callback(
    Output("crash-locations", "children"),
    [Input("data-sources", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_map(sources, start_date, end_date):
    """Update crash locations on the map."""
    if not sources:
        return []
    
    try:
        # Convert dates
        start = datetime.strptime(start_date.split('T')[0], '%Y-%m-%d') if start_date else None
        end = datetime.strptime(end_date.split('T')[0], '%Y-%m-%d') if end_date else None
        
        # Get data for all selected sources
        markers = []
        for source in sources:
            df = collector.get_source_data(source, start, end)
            if not df.empty and 'location' in df.columns:
                for idx, row in df.iterrows():
                    coords = get_coordinates(row['location'])
                    if coords:
                        # Create marker with popup
                        popup_text = f"""
                            <b>Date:</b> {row['date'].strftime('%Y-%m-%d')}<br>
                            <b>Location:</b> {row['location']}<br>
                            <b>Aircraft:</b> {row['aircraft_type']}<br>
                            <b>Severity:</b> {row['severity']}<br>
                            <b>Source:</b> {row['source']}
                        """
                        markers.append(
                            dl.Marker(
                                position=coords,
                                children=[
                                    dl.Popup(html.Div([
                                        html.P(popup_text)
                                    ]))
                                ]
                            )
                        )
        
        return markers
        
    except Exception as e:
        logger.error(f"Error updating map: {e}", exc_info=True)
        return []

@app.callback(
    Output("risk-output", "children"),
    [Input("analyze-button", "n_clicks")],
    [State("incident-description", "value")]
)
def predict_risk(n_clicks, description):
    """Predict risk based on incident description."""
    if n_clicks is None or not description:
        return ""
    
    try:
        # Placeholder for risk prediction
        risk_level = "Medium"  # This would come from your actual model
        confidence = 0.75
        
        return html.Div([
            html.H4("Risk Assessment Results"),
            html.P(f"Risk Level: {risk_level}"),
            html.P(f"Confidence: {confidence:.2%}"),
            html.Div([
                html.H5("Contributing Factors"),
                html.Ul([
                    html.Li("Weather conditions"),
                    html.Li("Time of day"),
                    html.Li("Aircraft type")
                ])
            ])
        ], className="prediction-results-content")
        
    except Exception as e:
        logger.error(f"Error predicting risk: {e}", exc_info=True)
        return html.Div("Error processing risk prediction")

def empty_figure(message):
    """Create an empty figure with a message."""
    return go.Figure().add_annotation(
        text=message,
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5
    )

def create_trend_analysis(df):
    """Create trend analysis plots"""
    if df.empty:
        return empty_figure("No data available for trend analysis")
    
    # Create subplot with shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Monthly Incident Count', 'Severity Distribution Over Time')
    )
    
    # Monthly incident count
    monthly_counts = df.resample('M')['date'].count()
    fig.add_trace(
        go.Scatter(
            x=monthly_counts.index,
            y=monthly_counts.values,
            mode='lines+markers',
            name='Monthly Incidents',
            line=dict(color='#1f77b4'),
            hovertemplate='Date: %{x}<br>Incidents: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Severity distribution over time
    severity_over_time = df.pivot_table(
        index=pd.Grouper(key='date', freq='M'),
        columns='severity',
        aggfunc='size',
        fill_value=0
    )
    
    colors = {'Fatal': '#e41a1c', 'Serious': '#ff7f00', 'Minor': '#4daf4a'}
    for severity in severity_over_time.columns:
        fig.add_trace(
            go.Scatter(
                x=severity_over_time.index,
                y=severity_over_time[severity],
                name=severity,
                mode='lines',
                line=dict(color=colors.get(severity, '#999999')),
                stackgroup='one'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig

def create_aircraft_analysis(df):
    """Create aircraft type analysis"""
    if df.empty:
        return empty_figure("No data available for aircraft analysis")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Aircraft Type Distribution',
            'Operator Analysis',
            'Phase of Flight',
            'Weather Conditions'
        )
    )
    
    # Aircraft Type Distribution
    aircraft_counts = df['aircraft_type'].value_counts().head(10)
    fig.add_trace(
        go.Bar(
            x=aircraft_counts.values,
            y=aircraft_counts.index,
            orientation='h',
            name='Aircraft Types',
            marker_color='#1f77b4'
        ),
        row=1, col=1
    )
    
    # Operator Analysis
    operator_counts = df['operator'].value_counts().head(10)
    fig.add_trace(
        go.Bar(
            x=operator_counts.values,
            y=operator_counts.index,
            orientation='h',
            name='Operators',
            marker_color='#2ca02c'
        ),
        row=1, col=2
    )
    
    # Phase of Flight
    phase_counts = df['phase_of_flight'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=phase_counts.index,
            values=phase_counts.values,
            name='Phase of Flight',
            hole=0.4
        ),
        row=2, col=1
    )
    
    # Weather Conditions
    weather_counts = df['weather_condition'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=weather_counts.index,
            values=weather_counts.values,
            name='Weather',
            hole=0.4
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output("trend-analysis", "figure"),
    [Input("data-sources", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_trend_analysis(sources, start_date, end_date):
    if not sources:
        return empty_figure("Please select data sources")
    
    try:
        df = collector.get_data(sources, start_date, end_date)
        return create_trend_analysis(df)
    except Exception as e:
        logger.error(f"Error creating trend analysis: {e}", exc_info=True)
        return empty_figure(f"Error: {str(e)}")

@app.callback(
    Output("aircraft-analysis", "figure"),
    [Input("data-sources", "value"),
     Input("date-range", "start_date"),
     Input("date-range", "end_date")]
)
def update_aircraft_analysis(sources, start_date, end_date):
    if not sources:
        return empty_figure("Please select data sources")
    
    try:
        df = collector.get_data(sources, start_date, end_date)
        return create_aircraft_analysis(df)
    except Exception as e:
        logger.error(f"Error creating aircraft analysis: {e}", exc_info=True)
        return empty_figure(f"Error: {str(e)}")

def create_feature_importance_plot(model, feature_names):
    """Create SHAP feature importance plot"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model.feature_matrix)
    
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values).mean(axis=0)
    
    # Calculate mean absolute SHAP values for feature importance
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=feature_importance_df['Importance'],
        y=feature_importance_df['Feature'],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Feature Importance Analysis',
        xaxis_title='Mean |SHAP value|',
        template='plotly_white',
        height=400
    )
    
    return fig

def find_similar_cases(input_vector, historical_vectors, historical_data, top_n=5):
    """Find similar historical cases using cosine similarity"""
    similarities = cosine_similarity(input_vector.reshape(1, -1), historical_vectors)
    similar_indices = similarities[0].argsort()[-top_n:][::-1]
    
    similar_cases = []
    for idx in similar_indices:
        case = historical_data.iloc[idx]
        similarity = similarities[0][idx]
        similar_cases.append({
            'date': case['date'].strftime('%Y-%m-%d'),
            'description': case['description'],
            'severity': case['severity'],
            'similarity': f"{similarity:.2%}"
        })
    
    return similar_cases

@app.callback(
    [Output('risk-score', 'children'),
     Output('confidence-score', 'children'),
     Output('feature-importance-plot', 'figure'),
     Output('similar-cases', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [State('incident-description', 'value'),
     State('aircraft-type-input', 'value'),
     State('weather-input', 'value'),
     State('flight-phase-input', 'value')]
)
def predict_risk(n_clicks, description, aircraft_type, weather, flight_phase):
    if not n_clicks or not description:
        raise dash.exceptions.PreventUpdate
    
    try:
        # Get prediction and confidence
        prediction, confidence, feature_vector = model.predict_with_confidence(
            description, aircraft_type, weather, flight_phase
        )
        
        # Create risk score display
        risk_display = html.Div([
            html.H3("Risk Level:", style={'margin-bottom': '10px'}),
            html.Div(
                prediction,
                className=f"risk-level {prediction.lower()}"
            )
        ])
        
        # Create confidence score display
        confidence_display = html.Div([
            html.H3("Confidence Score:", style={'margin-bottom': '10px'}),
            html.Div(
                f"{confidence:.1%}",
                className="confidence-level"
            )
        ])
        
        # Create feature importance plot
        importance_plot = create_feature_importance_plot(
            model.classifier,
            model.feature_names
        )
        
        # Find similar cases
        similar = find_similar_cases(
            feature_vector,
            model.historical_vectors,
            model.historical_data
        )
        
        # Create similar cases display
        similar_cases_display = html.Div([
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Date"),
                        html.Th("Description"),
                        html.Th("Severity"),
                        html.Th("Similarity")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(case['date']),
                        html.Td(case['description']),
                        html.Td(case['severity']),
                        html.Td(case['similarity'])
                    ]) for case in similar
                ])
            ], className="similar-cases-table")
        ])
        
        return risk_display, confidence_display, importance_plot, similar_cases_display
        
    except Exception as e:
        logger.error(f"Error predicting risk: {e}", exc_info=True)
        return (
            html.Div("Error processing risk prediction", style={'color': 'red'}),
            html.Div(""),
            empty_figure("Error generating feature importance"),
            html.Div("Error finding similar cases", style={'color': 'red'})
        )

@app.callback(
    [Output('aircraft-filter', 'options'),
     Output('operator-filter', 'options')],
    [Input('data-sources', 'value')]
)
def update_filter_options(sources):
    if not sources:
        return [], []
    
    try:
        df = load_crash_data(sources)
        
        aircraft_options = [{'label': t, 'value': t} 
                          for t in sorted(df['aircraft_type'].unique())]
        operator_options = [{'label': o, 'value': o} 
                          for o in sorted(df['operator'].unique())]
        
        return aircraft_options, operator_options
    except Exception as e:
        logger.error(f"Error updating filter options: {e}", exc_info=True)
        return [], []

@app.callback(
    Output('incident-table', 'data'),
    [Input('data-sources', 'value'),
     Input('severity-filter', 'value'),
     Input('aircraft-filter', 'value'),
     Input('operator-filter', 'value'),
     Input('search-input', 'value')]
)
def update_table(sources, severity, aircraft, operator, search):
    if not sources:
        return []
    
    try:
        df = load_crash_data(sources)
        
        # Apply filters
        if severity:
            df = df[df['severity'].isin(severity)]
        if aircraft:
            df = df[df['aircraft_type'].isin(aircraft)]
        if operator:
            df = df[df['operator'].isin(operator)]
        if search:
            df = df[df['description'].str.contains(search, case=False, na=False)]
        
        # Format date for display
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error updating table: {e}", exc_info=True)
        return []

@app.callback(
    [Output("incident-details-modal", "is_open"),
     Output("incident-details-body", "children")],
    [Input('incident-table', 'selected_rows'),
     Input('incident-table', 'data')],
    [State("incident-details-modal", "is_open")]
)
def show_incident_details(selected_rows, data, is_open):
    if not selected_rows:
        return False, None
    
    try:
        row = data[selected_rows[0]]
        
        details = html.Div([
            html.H4(f"Incident on {row['date']}"),
            html.Div([
                html.Strong("Location: "), 
                html.Span(row['location'])
            ]),
            html.Div([
                html.Strong("Severity: "), 
                html.Span(row['severity'])
            ]),
            html.Div([
                html.Strong("Aircraft: "), 
                html.Span(row['aircraft_type'])
            ]),
            html.Div([
                html.Strong("Operator: "), 
                html.Span(row['operator'])
            ]),
            html.Div([
                html.Strong("Description: "), 
                html.P(row['description'])
            ], className="incident-description")
        ])
        
        return True, details
    except Exception as e:
        logger.error(f"Error showing incident details: {e}", exc_info=True)
        return False, html.Div("Error loading incident details")

@app.callback(
    Output("download-link-container", "children"),
    [Input("export-button", "n_clicks")],
    [State('data-sources', 'value'),
     State('severity-filter', 'value'),
     State('aircraft-filter', 'value'),
     State('operator-filter', 'value'),
     State('search-input', 'value')]
)
def export_data(n_clicks, sources, severity, aircraft, operator, search):
    if not n_clicks or not sources:
        return None
    
    try:
        df = load_crash_data(sources)
        
        # Apply filters
        if severity:
            df = df[df['severity'].isin(severity)]
        if aircraft:
            df = df[df['aircraft_type'].isin(aircraft)]
        if operator:
            df = df[df['operator'].isin(operator)]
        if search:
            df = df[df['description'].str.contains(search, case=False, na=False)]
        
        return create_download_link(df, "crash_data.xlsx")
    except Exception as e:
        logger.error(f"Error exporting data: {e}", exc_info=True)
        return html.Div("Error exporting data", style={'color': 'red'})

def generate_excel_download(df):
    """Generate Excel file for download"""
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Crash Data', index=False)
    writer.close()
    output.seek(0)
    return base64.b64encode(output.read()).decode()

def create_download_link(df, filename):
    """Create download link for dataframe"""
    excel_file = generate_excel_download(df)
    return html.A(
        'Download Excel File',
        href=f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_file}",
        download=filename,
        className="download-link"
    )

if __name__ == "__main__":
    app.run_server(debug=True, threaded=True)
