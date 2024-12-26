import pytest
import os
from pathlib import Path

@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Set up test environment with necessary directories."""
    # Create test directories
    test_dirs = [
        tmp_path / "test_data",
        tmp_path / "test_models",
        tmp_path / "figures"
    ]
    
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for testing
    os.environ["TEST_DATA_DIR"] = str(test_dirs[0])
    os.environ["TEST_MODELS_DIR"] = str(test_dirs[1])
    os.environ["TEST_FIGURES_DIR"] = str(test_dirs[2])
    
    yield
    
    # Cleanup (optional, as tmp_path is automatically cleaned up)
    for dir_path in test_dirs:
        if dir_path.exists():
            for file in dir_path.glob("*"):
                file.unlink()
            dir_path.rmdir()
