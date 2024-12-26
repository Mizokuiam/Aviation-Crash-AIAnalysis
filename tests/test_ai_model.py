import pytest
import torch
import pandas as pd
from pathlib import Path
from src.models.ai_model import AIModel, TextEncoder, RiskPredictor

@pytest.fixture
def model():
    return AIModel(model_dir="tests/test_models")

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'description': [
            "Aircraft crashed during takeoff in heavy rain",
            "Engine failure led to emergency landing",
            "Bird strike caused loss of control"
        ],
        'severity': [0.8, 0.5, 0.6],
        'risk_factor': [0.7, 0.4, 0.5]
    })

def test_text_encoder():
    encoder = TextEncoder(vocab_size=100, embedding_dim=32, hidden_dim=64)
    batch_size = 2
    seq_length = 10
    x = torch.randint(0, 100, (batch_size, seq_length))
    output = encoder(x)
    assert output.shape == (batch_size, 64)

def test_risk_predictor():
    predictor = RiskPredictor(input_dim=64, hidden_dim=32, output_dim=2)  # Match the number of target columns
    batch_size = 2
    x = torch.randn(batch_size, 64)
    output = predictor(x)
    assert output.shape == (batch_size, 2)  # Two outputs: severity and risk_factor

def test_preprocess_text(model, sample_data):
    texts = sample_data['description'].tolist()
    encoded = model.preprocess_text(texts)
    assert isinstance(encoded, torch.Tensor)
    assert len(encoded) == len(texts)

def test_train(model, sample_data):
    history = model.train(
        train_data=sample_data,
        text_col='description',
        target_cols=['severity', 'risk_factor'],
        epochs=2,
        batch_size=2
    )
    assert 'loss' in history
    assert len(history['loss']) == 2

def test_predict(model, sample_data):
    # First train the model
    model.train(
        train_data=sample_data,
        text_col='description',
        target_cols=['severity', 'risk_factor'],
        epochs=2
    )
    
    # Then make predictions
    new_texts = ["Aircraft experienced turbulence during landing"]
    predictions = model.predict(new_texts, ['severity', 'risk_factor'])
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == 1
    assert list(predictions.columns) == ['severity', 'risk_factor']

def test_save_load_models(model, sample_data, tmp_path):
    # Train and save models
    model.train(
        train_data=sample_data,
        text_col='description',
        target_cols=['severity', 'risk_factor'],
        epochs=2
    )
    model.save_models()
    
    # Load models and make predictions
    model.load_models()
    new_texts = ["Aircraft experienced turbulence during landing"]
    predictions = model.predict(new_texts, ['severity', 'risk_factor'])
    assert isinstance(predictions, pd.DataFrame)
    assert not predictions.empty
