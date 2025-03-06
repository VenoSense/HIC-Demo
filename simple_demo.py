import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import os
import base64
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="DVT Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for styling ---
st.markdown("""
<style>
    /* Global styling */
    body {
        font-family: 'Inter', sans-serif;
    }
    
    /* Card styling */
    .info-card {
        background-color: #f0f7ff;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        margin-bottom: 18px;
        border: 1px solid #e6f0ff;
    }
    
    /* Card header styling */
    .card-header {
        color: #0a2540;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
        border-bottom: 1px solid #d0e0ff;
        padding-bottom: 8px;
    }
    
    /* Alert styling */
    .alert-danger {
        background-color: rgba(128,0,32,0.06);
        color: #800020;
        border-left: 5px solid #800020;
        padding: 15px;
        margin: 12px 0;
        border-radius: 6px;
    }
    
    /* Highlight color for important metrics */
    .highlight-value {
        color: #800020;
        font-weight: 600;
    }
    
    /* Secondary highlight for positive values */
    .positive-value {
        color: #008060;
        font-weight: 600;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-critical {
        background-color: #800020;
        box-shadow: 0 0 0 4px rgba(128,0,32,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("DVT Detection System")
st.markdown("<div style='background-color:#f0f7ff;padding:8px 15px;border-radius:8px;display:inline-block;'>Case ID: DVT-20231104-001</div>", unsafe_allow_html=True)

# --- Information About Simplified Mode ---
st.warning("""
This is a simplified version of the DVT Detection System that doesn't require OpenCV.
It works with Python 3.12 without compatibility issues.
""")

# --- Sidebar ---
with st.sidebar:
    st.subheader("Video Upload")
    uploaded_file = st.file_uploader("Upload your own video", type=["mp4"])
    
    if uploaded_file is not None:
        st.success("Video uploaded successfully!")
        
    st.subheader("Patient Information")
    st.markdown("""
    **Name:** John Doe  
    **DOB:** 05/12/1965  
    **ID:** 12345678  
    **Gender:** Male  
    **Weight:** 82 kg  
    **Height:** 178 cm  
    **Blood Type:** O+
    """)
    
    st.subheader("Scan Details")
    st.markdown("""
    **Scan Date:** 11/04/2023  
    **Physician:** Dr. Sarah Johnson  
    **Equipment:** Philips EPIQ Elite  
    **Department:** Vascular Imaging
    """)
    
    st.caption("**Disclaimer:** This is a demonstration system. Not for clinical use.")

# --- Main Layout ---
current_date = datetime.now().strftime("%m/%d/%Y")

# First row - Video and Alert
row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    st.markdown("<h3 style='text-align: center;'>Ultrasound with Detection</h3>", unsafe_allow_html=True)
    
    # Instead of using OpenCV, just display a placeholder image or embedded video
    st.markdown("""
    <div style="border: 2px solid #e6f0ff; border-radius:12px; overflow:hidden; text-align:center; background-color:#f5f9ff;">
        <img src="https://via.placeholder.com/640x480?text=Ultrasound+Video+Simulation" width="100%">
    </div>
    """, unsafe_allow_html=True)
    
    st.info("In this simplified version, video processing functionality is replaced with a static image.")

with row1_col2:
    # DVT Alert at top right
    st.markdown("""
    <div class="alert-danger">
        <h3><span class="status-indicator status-critical"></span> DVT ALERT: Clot Detected</h3>
        <p>Urgent clinical evaluation recommended</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status info below the alert
    st.markdown(f"""
    <div class='info-card'>
        <div class='card-header'>Patient & Scan Info</div>
        <p><strong>Date:</strong> {current_date}</p>
        <p><strong>Patient:</strong> John Doe</p>
        <p><strong>Last Scan:</strong> 02/05/2023 - No clot detected</p>
    </div>
    """, unsafe_allow_html=True)

# Second row - Analysis data
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.markdown("""
    <div class="info-card">
        <div class="card-header">Key Findings</div>
        <ul>
            <li><strong>Location:</strong> <span class="highlight-value">Proximal femoral vein</span></li>
            <li><strong>Size:</strong> 2.5 CM</li>
            <li><strong>Flow Disruption:</strong> Partial occlusion</li>
            <li><strong>Pattern:</strong> Non-compressible segment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with row2_col2:
    st.markdown("""
    <div class="info-card">
        <div class="card-header">Vessel Comparison</div>
        <table style="width:100%">
            <tr>
                <th>Metric</th>
                <th>Affected</th>
                <th>Normal</th>
            </tr>
            <tr>
                <td>Vessel Diameter</td>
                <td class="highlight-value">10.2 mm</td>
                <td class="positive-value">7.5 mm</td>
            </tr>
            <tr>
                <td>Flow Velocity</td>
                <td class="highlight-value">8.4 cm/s</td>
                <td class="positive-value">19.3 cm/s</td>
            </tr>
            <tr>
                <td>Augmentation</td>
                <td class="highlight-value">5%</td>
                <td class="positive-value">45%</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# Flow Analysis Chart
st.subheader("Flow Analysis")

# Create a simple dual-plot chart with Plotly
fig = make_subplots(rows=2, cols=1, 
                   shared_xaxes=True,
                   subplot_titles=("Normal Vein - Phasic Flow", "DVT Affected Vein - Diminished Flow"))

# Generate time points and sample data
x_time = np.linspace(0, 4, 100)
normal_data = 20 + 12 * np.sin(2*np.pi*x_time) + 5 * np.sin(6*np.pi*x_time) 
dvt_data = 10 + 4 * np.sin(2*np.pi*x_time) + np.random.normal(0, 1, 100)

fig.add_trace(go.Scatter(x=x_time, y=normal_data, mode='lines', name='Normal Vein'),
             row=1, col=1)
fig.add_trace(go.Scatter(x=x_time, y=dvt_data, mode='lines', name='DVT Affected Vein'),
             row=2, col=1)

fig.update_layout(height=400, title_text="Spectral Doppler Waveform Analysis")
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="Flow Velocity (cm/s)", row=1, col=1)
fig.update_yaxes(title_text="Flow Velocity (cm/s)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# Bottom row - Actions and Additional Data
row3_col1, row3_col2, row3_col3 = st.columns([1, 1, 1])

with row3_col1:
    st.markdown("""
    <div class="info-card">
        <div class="card-header">Risk Assessment</div>
        <ul>
            <li><span class="highlight-value">Wells Score: 6 (High)</span></li>
            <li>D-dimer: 1450 ng/mL (Elevated)</li>
            <li>Immobilization > 3 days</li>
            <li>Prior DVT history: No</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with row3_col2:
    st.markdown("""
    <div class="info-card">
        <div class="card-header">Clinical Recommendations</div>
        <ol>
            <li><span class="highlight-value">Urgent vascular consultation</span></li>
            <li>Consider anticoagulation therapy</li>
            <li>Confirmatory duplex ultrasound</li>
            <li>Monitor for PE symptoms</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

with row3_col3:
    st.markdown("""
    <div class="info-card">
        <div class="card-header">Previous Examinations</div>
        <p><strong>Current:</strong> <span class="highlight-value">DVT Detected</span></p>
        <p><strong>02/05/2023:</strong> <span class="positive-value">Normal Study</span></p>
        <p><strong>01/12/2023:</strong> <span class="positive-value">Normal Study</span></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("""
This is a simplified version without OpenCV dependencies, designed to work with Python 3.12. 
To run the full version with video processing, use Python 3.11 or earlier, or install the compatible packages.
""")
