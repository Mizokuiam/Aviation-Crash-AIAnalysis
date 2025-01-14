Last Modified: 2025-01-14

# API Version: main

# PyCrashAI API Documentation

## Table of Contents
- [Data Collection](#data-collection)
- [Data Analysis](#data-analysis)
- [AI Model](#ai-model)
- [Model Training](#model-training)
- [Automation](#automation)
- [Dashboard](#dashboard)

## Data Collection

### CrashDataCollector

```python
from src.collectors.crash_data_collector import CrashDataCollector
```

Class for collecting and managing aviation crash data.

#### Methods

##### `__init__(data_dir: str = "data/raw")`
Initialize the collector with a data directory.

##### `async def collect_data() -> None`
Asynchronously collect crash data from various sources.

##### `async def fetch_page(url: str) -> str`
Fetch HTML content from a URL.

##### `async def parse_crash_data(html: str) -> List[Dict[str, Any]]`
Parse crash data from HTML content.

##### `def load_latest_data() -> pd.DataFrame`
Load the most recent crash data file.

## Data Analysis

### DataAnalyzer

```python
from src.analysis.data_analyzer import DataAnalyzer
```

Class for analyzing crash data patterns and generating insights.

#### Methods

##### `__init__(data_dir: str = "data")`
Initialize the analyzer with a data directory.

##### `def load_data(filename: Optional[str] = None) -> pd.DataFrame`
Load crash data from file.

##### `def analyze_patterns() -> Dict[str, Any]`
Analyze patterns in crash data, including temporal, geographical, and causal factors.

##### `def generate_visualizations(output_dir: str) -> None`
Generate interactive visualizations of crash patterns.

## AI Model

### AIModel

```python
from src.models.ai_model import AIModel
```

Main AI model handler for crash analysis and predictions.

#### Methods

##### `__init__(model_dir: str = "models")`
Initialize the model with a model directory.

##### `def train(train_data: pd.DataFrame, text_col: str, target_cols: List[str], epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]`
Train the model on crash data.

##### `def predict(texts: List[str], target_cols: Optional[List[str]] = None) -> pd.DataFrame`
Make predictions for new crash scenarios.

##### `def save_models() -> None`
Save trained models and vocabularies.

##### `def load_models() -> None`
Load trained models and vocabularies.

## Model Training

### ModelTrainer

```python
from src.models.model_trainer import ModelTrainer
```

Enhanced model trainer with cross-validation and versioning.

#### Methods

##### `__init__(model_dir: str = "models")`
Initialize the trainer with a model directory.

##### `def train_with_cv(model: AIModel, train_data: pd.DataFrame, text_col: str, target_cols: List[str], n_splits: int = 5, epochs: int = 10, batch_size: int = 32) -> Dict`
Train model with k-fold cross-validation.

##### `def get_version_history() -> List[Dict]`
Get the history of model versions and their metrics.

##### `def get_current_version() -> Dict`
Get the current model version information.

## Automation

### AutomationMonitor

```python
from src.automation.monitor import AutomationMonitor
```

Monitor class for automated data collection and model retraining.

#### Methods

##### `__init__(data_dir: str = "data", model_dir: str = "models", config_file: str = "automation_config.json")`
Initialize the automation monitor.

##### `async def run() -> None`
Run the automation monitor continuously.

## Dashboard

### DashApp

```python
from src.dashboard.app import app
```

Dash application for interactive data visualization and model interaction.

#### Components

- Analysis Tab: Visualize temporal patterns, geographical distribution, and causal factors
- Risk Prediction Tab: Make real-time predictions for new scenarios
- Model Performance Tab: Monitor model metrics and version history

#### Usage

```python
if __name__ == "__main__":
    app.run_server(debug=True)
```
