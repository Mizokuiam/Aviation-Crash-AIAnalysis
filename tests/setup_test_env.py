from pathlib import Path

# Create test directories
test_dirs = [
    Path("test_data"),
    Path("test_models"),
    Path("figures")
]

for dir_path in test_dirs:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {dir_path}")
