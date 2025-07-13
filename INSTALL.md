# Installation Guide

This guide will help you set up the Mount Rainier Weather Prediction Tool on your system.

## Prerequisites

- **Python 3.8 or higher**
- **Git** (for cloning the repository)
- **Internet connection** (for downloading weather data and dependencies)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Weather_Prediction.git
cd Weather_Prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Test the Installation

```bash
python test_app.py
```

If all tests pass, you should see:
```
✅ All tests passed! The app should work correctly.
```

### 5. Run the Application

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

## Optional: API Keys

For full functionality, you may want to set up API keys:

### ERA5 Weather Data (Optional)

1. Register at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
2. Copy your API key
3. Create a `.env` file in the project root:

```bash
cp config/env_example.txt .env
```

4. Edit `.env` and add your API key:
```
CDS_API_KEY=your_api_key_here
```

## Troubleshooting

### Common Issues

**XGBoost installation fails on macOS:**
```bash
brew install libomp
pip install xgboost
```

**Streamlit not found:**
```bash
pip install streamlit
```

**Permission errors on Linux/macOS:**
```bash
chmod +x test_app.py
chmod +x setup.py
```

### Getting Help

If you encounter issues:

1. Check that all dependencies are installed: `pip list`
2. Run the test script: `python test_app.py`
3. Check the logs for specific error messages
4. Ensure you're using Python 3.8 or higher: `python --version`

## Development Setup

For developers who want to contribute:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python test_app.py

# Format code
black src/ app/

# Check code quality
flake8 src/ app/
```

## Data Sources

The application uses multiple data sources:

- **Open-Meteo API**: Free weather data (no API key required)
- **ERA5 Satellite Data**: High-quality historical data (requires free registration)
- **Camp Muir Weather Station**: Local ground truth data

## Support

For questions or issues:
- Check the [README.md](README.md) for detailed documentation
- Review the [troubleshooting section](#troubleshooting)
- Open an issue on GitHub

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 