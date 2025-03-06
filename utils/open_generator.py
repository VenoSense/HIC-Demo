import webbrowser
import os
import sys
import time

def main():
    """Open the heatmap generator page in the default web browser."""
    # Determine the Streamlit app URL
    # Default is localhost:8501 but this may change if multiple apps are running
    base_url = "http://localhost:8502"
    generator_url = f"{base_url}/?page=generate"
    
    print("Opening heatmap video generator...")
    print(f"URL: {generator_url}")
    
    # Try to open the URL in the default browser
    try:
        webbrowser.open(generator_url)
        print("Browser opened successfully!")
    except Exception as e:
        print(f"Error opening browser: {e}")
        print(f"Please manually navigate to: {generator_url}")
    
    print("\nMake sure the Streamlit app is running!")
    print("To start the app, run: streamlit run app.py")

if __name__ == "__main__":
    main()
