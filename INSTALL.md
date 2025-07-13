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

## Data Sources

The application uses multiple data sources:

- **Open-Meteo API**: Free weather data (no API key required)
- **ERA5 Satellite Data**: High-quality historical data (requires free registration)
- **Camp Muir Weather Station**: Local ground truth data

## Support

For questions or issues:
- Check the [README.md](README.md) for detailed documentation
- Open an issue on GitHub

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
