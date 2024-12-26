# Aviation Crash Analysis Dashboard

A comprehensive dashboard for analyzing aviation crash data with AI-powered risk prediction and interactive visualizations.

## Features

- **Data Collection & Integration**
  - Multi-source data collection (NTSB, FAA, ASN)
  - Automated data updates
  - MongoDB integration for efficient storage
  - Data standardization and cleaning

- **Interactive Dashboard**
  - Real-time crash data visualization
  - Advanced filtering options
  - Trend analysis
  - Aircraft type and operator statistics
  - Export functionality for filtered data

- **AI-Powered Analysis**
  - Risk prediction for incidents
  - Feature importance analysis
  - Confidence scoring
  - Natural language processing for crash descriptions

- **Performance Optimizations**
  - Multi-level caching system
  - Database indexing
  - Efficient query optimization
  - Background processing for long-running tasks

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pycrashai.git
   cd pycrashai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up MongoDB:
   - Install MongoDB Community Edition
   - Start MongoDB service
   - Default connection: mongodb://localhost:27017/

## Project Structure

```
pycrashai/
├── data/               # Data storage
│   └── raw/           # Raw collected data
├── models/            # Trained ML models
├── src/
│   ├── analysis/      # Data analysis modules
│   ├── collectors/    # Data collection modules
│   ├── dashboard/     # Dashboard application
│   ├── database/      # Database operations
│   └── models/        # AI model definitions
├── tests/             # Unit tests
└── requirements.txt   # Project dependencies
```

## Usage

1. Start the dashboard:
   ```bash
   python src/dashboard/app.py
   ```

2. Access the dashboard at `http://127.0.0.1:8050/`

3. Available features:
   - View crash statistics and trends
   - Filter data by date, severity, aircraft type, etc.
   - Export filtered data to Excel
   - Analyze risk factors using AI predictions
   - View geographical distribution of incidents

## Configuration

- Environment variables can be set in `.env`:
  ```
  MONGODB_URI=mongodb://localhost:27017/
  DATA_DIR=data/raw
  MODEL_DIR=models
  ```

## Performance Features

1. **Caching System**:
   - Source data cache (5 min TTL)
   - Query results cache (1 min TTL)
   - Statistics cache (5 min TTL)

2. **Database Optimization**:
   - Indexed fields: date, source, severity, aircraft_type, operator
   - Text search capabilities for descriptions
   - Efficient aggregation pipelines

3. **UI Optimizations**:
   - Lazy loading of components
   - Pagination for large datasets
   - Background processing for computations

## Development

1. Run tests:
   ```bash
   pytest tests/
   ```

2. Code style:
   ```bash
   flake8 src/
   black src/
   ```

## Dependencies

- **Core**:
  - Python 3.11+
  - MongoDB 4.4+
  - Dash 2.14.1
  - Pandas 2.1.4
  - Scikit-learn 1.3.2

- **AI/ML**:
  - TensorFlow 2.18.0
  - SHAP 0.44.0
  - Spacy 3.8.3

- **Data Processing**:
  - NumPy 1.26.2
  - Plotly 5.18.0
  - XlsxWriter 3.1.9

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NTSB for providing aviation accident data
- FAA for regulatory information
- Aviation Safety Network for additional data sources
