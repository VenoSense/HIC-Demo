#!/bin/bash
# Setup script to install necessary dependencies for the DVT Demo

echo "Setting up DVT Demo environment..."

# Check Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Using Python $PYTHON_VERSION"

# For Python 3.12+, we need to handle special dependencies
if [[ $PYTHON_VERSION == 3.12* ]] || [[ $PYTHON_VERSION == 3.13* ]]; then
    echo "Python 3.12+ detected - using compatible dependencies"
    echo "numpy>=1.26.0" > requirements_py312.txt
    echo "streamlit==1.43.0" >> requirements_py312.txt
    echo "pandas>=2.0.1" >> requirements_py312.txt
    echo "plotly>=5.14.1" >> requirements_py312.txt
    echo "jinja2>=3.1.2" >> requirements_py312.txt
    echo "python-dateutil>=2.8.2" >> requirements_py312.txt
    echo "setuptools>=65.5.0" >> requirements_py312.txt
    
    pip install -r requirements_py312.txt
    
    echo "Installing lightweight dependencies for Python 3.12+ compatibility"
    echo "Note: OpenCV will not be installed - using simplified version."
else
    # For Python 3.11 and earlier, install the full dependencies
    echo "Python 3.11 or earlier detected - installing full dependencies"
    pip install -r requirements.txt
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p videos
mkdir -p assets

echo "Setup complete!"
echo "If you're on Python 3.12+, run with: streamlit run simple_demo.py"
echo "For older Python versions: streamlit run demo.py"
