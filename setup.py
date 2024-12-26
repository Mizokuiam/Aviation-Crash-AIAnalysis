from setuptools import setup, find_packages

setup(
    name="pycrashai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "spacy",
        "numpy",
        "pandas",
        "scikit-learn",
        "requests",
        "beautifulsoup4",
        "plotly",
        "dash",
        "python-dotenv",
        "pytest",
        "pytest-asyncio",
        "jupyter",
        "aiohttp"
    ]
)
