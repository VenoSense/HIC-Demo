# DVT Detection System Demo

This repository contains a demonstration of an interactive ultrasound DVT (Deep Vein Thrombosis) detection system.

## Python Version Compatibility

- **Python 3.11 or earlier**: Full functionality with video processing
- **Python 3.12 or newer**: Simplified version without OpenCV (Python 3.12 removed `distutils` which older NumPy/OpenCV versions require)

## Installation

1. **Setup environment:**

```bash
# Option 1: Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Option 2: Use the setup script (Linux/Mac)
chmod +x setup.sh
./setup.sh
```

2. **Install dependencies manually:**

```bash
# For Python 3.11 and earlier:
pip install -r requirements.txt

# For Python 3.12+:
pip install streamlit numpy>=1.26.0 pandas plotly jinja2 python-dateutil
```

## Running the Demo

```bash
# For Python 3.11 and earlier (full feature version):
streamlit run demo.py

# For Python 3.12+ (simplified version):
streamlit run simple_demo.py
```

## Troubleshooting

If you encounter issues with OpenCV or NumPy on Python 3.12+:

1. Use the simplified version: `streamlit run simple_demo.py`
2. Downgrade to Python 3.11
3. Try installing with: `pip install opencv-python-headless --no-binary opencv-python-headless`

## Structure

- `demo.py`: Main application with full video processing
- `simple_demo.py`: Simplified version without OpenCV dependencies
- `requirements.txt`: Dependencies for full version
- `videos/`: Directory for video files
- `assets/`: Directory for images and other assets

## License

This project is provided as an educational demonstration.
