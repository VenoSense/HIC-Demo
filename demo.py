import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import os
from pathlib import Path
import tempfile
import io
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="HIC Ultrasound DVT Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for styling with modern color scheme ---
st.markdown("""
<style>
    /* Global styling */
    @import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap');
    
    /* Main container styling */
    body {
        font-family: 'Ubuntu', sans-serif;
    }
    
    .main {
        background-color: rgba(255, 255, 255, 1);
        padding-top: 0 !important;
    }
    
    /* Reduce padding at the top to accommodate banner */
    .block-container {
        padding-top: 1rem !important;
        margin-top: 0 !important;
    }
    
    /* Card styling - light blue with rounded corners */
    .info-card {
        background-color: #f0f7ff;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        margin-bottom: 15px;
        border: 1px solid #e6f0ff;
        transition: all 0.2s ease;
    }
    
    .info-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    /* Card header styling - deep blue text */
    .card-header {
        color: #0a2540;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 8px;
        border-bottom: 1px solid #d0e0ff;
        padding-bottom: 6px;
        letter-spacing: 0.01em;
    }
    
    /* Alert styling - maroon for DVT alert */
    .alert-danger {
        background-color: rgba(128,0,32,0.06);
        color: #800020;
        border-left: 5px solid #800020;
        padding: 10px;
        margin: 10px 0;
        border-radius: 6px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(128,0,32,0.4);
        }
        70% {
            box-shadow: 0 0 0 8px rgba(128,0,32,0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(128,0,32,0);
        }
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
    
    /* Metrics styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .metric-item {
        flex: 1 1 48%;
        background-color: white; /* Changed to white */
        padding: 8px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e6f0ff;
    }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #0a2540;
        margin-bottom: 2px;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6080a0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Logo container - hide since we're using banner */
    .logo-container {
        display: none;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 36px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding: 6px 14px;
        font-weight: 400;
        color: #0a2540;
    }

    .stTabs [aria-selected="true"] {
        background-color: #f0f7ff;
        font-weight: 600;
        border-bottom: 2px solid #0068c9;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-critical {
        background-color: #800020;
        box-shadow: 0 0 0 4px rgba(128,0,32,0.2);
    }
    
    .status-warning {
        background-color: #FFD700;
        box-shadow: 0 0 0 4px rgba(255,215,0,0.2);
    }
    
    .status-normal {
        background-color: #008060;
        box-shadow: 0 0 0 4px rgba(0,128,96,0.2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #0068c9;
    }
    
    /* Sidebar styling - light blue instead of white */
    .css-1d391kg {
        background-color: #f0f7ff;
    }
    
    /* Video player area styling */
    .video-display {
        border: 2px solid #e6f0ff;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-top: 0 !important;
    }
    
    /* Video controls container */
    .video-controls {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 8px;
        margin-top: 8px;
        border: 1px solid #e6f0ff;
        display: flex;
        justify-content: center;
        gap: 6px;
    }
    
    /* Speed control buttons */
    .speed-button {
        background-color: #f0f7ff;
        border: 1px solid #d0e0ff;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .speed-button.active {
        background-color: #0068c9;
        color: white;
        border-color: #0068c9;
        box-shadow: 0 2px 5px rgba(0,104,201,0.3);
    }
    
    .speed-button:hover:not(.active) {
        background-color: #f0f7ff;
        transform: translateY(-1px);
    }
    
    /* Table styling */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    
    th {
        background-color: #f0f7ff;
        padding: 8px;
        text-align: left;
        color: #0a2540;
        font-weight: 600;
        border-bottom: 1px solid #d0e0ff;
    }
    
    td {
        padding: 8px;
        border-bottom: 1px solid #e6f0ff;
        color: #0a2540;
    }
    
    /* Footer styling */
    footer {
        visibility: hidden;
    }
    
    /* Hide hamburger menu */
    #MainMenu {
        visibility: hidden;
    }
    
    /* Headings styling - more compact */
    h1, h2, h3, h4, h5, h6 {
        color: #0a2540;
        font-weight: 600;
        margin-top: 0.5em !important;
        margin-bottom: 0.5em !important;
    }
    
    h1 {
        font-size: 1.6rem;
    }
    
    h3 {
        font-size: 1.2rem;
        margin-bottom: 12px;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    
    /* Video file selector styling */
    div[data-testid="stSelectbox"] {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Case file info badge */
    .case-badge {
        display: inline-block;
        background-color: #f0f7ff;
        color: #0068c9;
        border-radius: 15px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 5px 0;
        border: 1px solid #d0e0ff;
    }
    
    /* Make alerts and notices more visible */
    .stAlert {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    
    .dvt-heat-zones {
        position: relative;
    }
    
    .heat-indicator {
        position: absolute;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background-color: rgba(255,0,0,0.6);
        filter: blur(10px);
        box-shadow: 0 0 15px 5px rgba(255,0,0,0.6);
        animation: pulse-red 2s infinite;
        z-index: 100;
    }
    
    @keyframes pulse-red {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(255,0,0,0.7);
        }
        
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(255,0,0,0);
        }
        
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(255,0,0,0);
        }
    }
    
    /* Reduce spacing everywhere */
    .element-container {
        margin-bottom: 0.5em !important;
    }
    
    /* Make sidebar more compact */
    .css-hxt7ib {
        padding-top: 2rem !important;
    }
    
    /* Reduce padding for streamlit components */
    .stMarkdown, .stText {
        margin-top: 0 !important;
        margin-bottom: 0.5em !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* More compact text */
    p {
        margin-bottom: 0.5em;
        line-height: 1.4;
    }
    
    /* More compact lists */
    ul, ol {
        padding-left: 1.5em;
        margin-top: 0.3em;
        margin-bottom: 0.5em;
    }
    
    li {
        margin-bottom: 0.2em;
    }
    
    /* Update plot background to light blue but preserve chart visibility */
    .js-plotly-plot .plotly .main-svg {
        background-color: rgba(240,247,255,0.3) !important;
    }
    
    /* Ensure plot elements remain visible */
    .js-plotly-plot .plotly .main-svg .plot-container {
        opacity: 1 !important;
    }
    
    /* Make sure grid lines and axes are visible */
    .js-plotly-plot .plotly .main-svg .gridlayer path,
    .js-plotly-plot .plotly .main-svg .axislayer path,
    .js-plotly-plot .plotly .main-svg text {
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_available_videos():
    """Get a list of all available video files in the videos directory."""
    # Check if there's an uploaded video in the session state
    if 'uploaded_video' in st.session_state and st.session_state.uploaded_video is not None:
        return ["uploaded_video.mp4"]
    
    # Otherwise use default video if available
    video_dir = "videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)
        return []
    
    videos = [f.name for f in Path(video_dir).glob("*.mp4")]
    return videos if videos else ["placeholder.mp4"]

def create_placeholder_video():
    """Create a placeholder video if no videos are available."""
    placeholder_path = os.path.join("videos", "placeholder.mp4")
    
    if os.path.exists(placeholder_path):
        return placeholder_path
        
    # Create a placeholder video
    height, width = 480, 640
    fps = 30
    seconds = 5
    
    os.makedirs("videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(placeholder_path, fourcc, fps, (width, height))
    
    # Create and write frames
    for i in range(fps * seconds):
        # Create a blank frame with frame number
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some gradient for visual interest
        for y in range(height):
            for x in range(width):
                frame[y, x, 0] = int(255 * y / height)  # Blue channel
                frame[y, x, 1] = int(255 * x / width)   # Green channel
                frame[y, x, 2] = 128  # Red channel
                
        # Add text
        cv2.putText(frame, "No videos available", (width//4, height//2-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame {i+1}/{fps*seconds}", (width//4, height//2+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    return placeholder_path

# --- Display Logo ---
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("assets/logo.png", width=120)
st.markdown('</div>', unsafe_allow_html=True)

# --- App Header ---
st.markdown("<div class='case-badge'>Case ID: DVT-20231104-001</div>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.subheader("Video Upload")
    
    # File uploader for MP4 videos
    uploaded_file = st.file_uploader("Upload your own video", type=["mp4"])
    
    if uploaded_file is not None:
        # Save the uploaded video to session state
        st.session_state.uploaded_video = uploaded_file.read()
        st.success("Video uploaded successfully!")
        
        # Add button to remove uploaded video
        if st.button("Remove Uploaded Video"):
            st.session_state.uploaded_video = None
            st.rerun()
    
    # Disclaimer about uploading videos
    if uploaded_file is None:
        st.info("Upload your own ultrasound video for analysis. The video will not be stored permanently.")
    
    # Patient info - no columns in sidebar
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
    **Technician:** Robert Williams  
    **Equipment:** Philips EPIQ Elite  
    **Department:** Vascular Imaging
    """)
    
    # Disclaimer
    st.markdown("---")
    st.caption("**Disclaimer:** This is a demonstration system. Not for clinical use.")

# --- Main Layout ---
# Current date for display
current_date = datetime.now().strftime("%m/%d/%Y")

# First row - Video and Alert
row1_col1, row1_col2 = st.columns([2, 1])

# Top Left - Video Display
with row1_col1:
    try:
        st.markdown('<div class="video-display">', unsafe_allow_html=True)
        
        # Use the specified video path with error checking
        video_path = "videos/venosensedemo.mp4"
        
        # Check if the video file exists
        if os.path.exists(video_path):
            # Display the specified video
            st.video(
                video_path,
                format="video/mp4",
                start_time=0,
                loop=True,
                autoplay=True,
                end_time=None
            )
        else:
            # Check for any MP4 file in the videos directory
            video_dir = "videos"
            if not os.path.exists(video_dir):
                os.makedirs(video_dir, exist_ok=True)
                
            other_videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            
            if other_videos:
                alternative_video = os.path.join(video_dir, other_videos[0])
                st.warning(f"Video 'venosensedemo.mp4' not found. Using {other_videos[0]} instead.")
                st.video(
                    alternative_video,
                    format="video/mp4",
                    start_time=0,
                    loop=True,
                    autoplay=True,
                    end_time=None
                )
            else:
                st.warning("No video files found. Please add MP4 files to the 'videos' directory.")
                st.image("https://via.placeholder.com/640x480.png?text=Video+Not+Found", width=640)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Hide standard video controls with CSS if needed
        st.markdown("""
        <style>
            /* Remove extra padding and margins from video container */
            .video-display {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
            
            /* Reduce padding in the stVideo element */
            .element-container .stVideo {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
            
            /* Attempt to hide standard video controls (may not work in all browsers) */
            video::-webkit-media-controls {
                display: none !important;
            }
            
            /* Add custom play button overlay for touch devices */
            .video-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                pointer-events: none;
            }
            
            .play-button {
                width: 80px;
                height: 80px;
                background-color: rgba(0,0,0,0.5);
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
            }
        </style>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error playing video: {str(e)}")
        st.markdown("""
        <div style="padding: 20px; text-align: center; background-color: #f0f0f0; border-radius: 10px;">
            <h3>Video playback error</h3>
            <p>Unable to play videos/venosensedemo.mp4. Please ensure the file exists in the videos directory.</p>
        </div>
        """, unsafe_allow_html=True)

# Top Right - Alert and Status
with row1_col2:
    # DVT Alert at top right
    st.markdown("""
    <div class="alert-danger">
        <h3><span class="status-indicator status-critical"></span> DVT ALERT: Clot Detected</h3>
        <p>Urgent clinical evaluation recommended</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status info below the alert - without using nested columns
    st.markdown(f"""
    <div class='info-card'>
        <div class='card-header'>Patient & Scan Info</div>
        <table style="width:100%; border-collapse: separate; border-spacing: 0;">
            <tr>
                <td style="padding: 8px 4px; border-bottom: 1px solid #e6f0ff;"><strong>Date:</strong></td>
                <td style="padding: 8px 4px; border-bottom: 1px solid #e6f0ff;">{current_date}</td>
            </tr>
            <tr>
                <td style="padding: 8px 4px; border-bottom: 1px solid #e6f0ff;"><strong>Patient:</strong></td>
                <td style="padding: 8px 4px; border-bottom: 1px solid #e6f0ff;">John Doe (ID: 12345678)</td>
            </tr>
            <tr>
                <td style="padding: 8px 4px;"><strong>Previous Scan:</strong></td>
                <td style="padding: 8px 4px;">02/05/2023 - No clot detected</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# Second row - Key metrics
row2_col1, row2_col2 = st.columns([1, 2])

with row1_col2:
    st.markdown("""
    <div class="info-card">
        <div class="card-header">Detection Confidence</div>
        <div class="metric-container">
            <div class="metric-item">
                <div class="metric-value highlight-value">95%</div>
                <div class="metric-label">AI Confidence</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">High</div>
                <div class="metric-label">Certainty Level</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

# Third row - Vessel comparison table with modern design
row3_col1, row3_col2 = st.columns([1, 1])

with row2_col2:
    st.markdown("""
    <div class="info-card">
        <div class="card-header">Flow Analysis</div>
        <div class="metric-container">
            <div class="metric-item">
                <div class="metric-value highlight-value">8.4 cm/s</div>
                <div class="metric-label">Peak Systolic</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">0.5</div>
                <div class="metric-label">Resistance Index</div>
            </div>
        </div>
        <div class="metric-container" style="margin-top: 10px;">
            <div class="metric-item">
                <div class="metric-value highlight-value">4.2 cm/s</div>
                <div class="metric-label">End Diastolic</div>
            </div>
            <div class="metric-item">
                <div class="metric-value highlight-value">Absent</div>
                <div class="metric-label">Augmentation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Fourth row - Additional information
row4_col1, row4_col2, row4_col3 = st.columns([1, 1, 1])

with row4_col1:
    st.markdown("""
    <div class="info-card">
        <div class="card-header">Risk Assessment</div>
        <ul>
            <li><span class="highlight-value">Wells Score: 6 (High)</span></li>
            <li>D-dimer: 1450 ng/mL (Elevated)</li>
            <li>Immobilization > 3 days</li>
            <li>Prior DVT history: No</li>
            <li>Family history: Positive</li>
            <li>BMI: 25.9 (Overweight)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with row4_col2:
    st.markdown("""
    <div class="info-card">
        <div class="card-header">Clinical Recommendations</div>
        <ol>
            <li><span class="highlight-value">Urgent vascular consultation recommended</span></li>
            <li>Consider anticoagulation therapy</li>
            <li>Confirmatory duplex ultrasound</li>
            <li>Monitor for pulmonary embolism symptoms</li>
            <li>Consider IVC filter if anticoagulation contraindicated</li>
            <li>Follow-up scan in 2-4 weeks</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

with row4_col3:
    st.markdown("""
    <div class="info-card">
        <div class="card-header">Previous Examinations</div>
        <div style="font-size: 0.9em;">
            <p><strong>Current:</strong> <span class="highlight-value">DVT Detected</span></p>
            <p><strong>02/05/2023:</strong> <span class="positive-value">Normal Study</span></p>
            <p><strong>01/12/2023:</strong> <span class="positive-value">Normal Study</span></p>
            <p><strong>10/18/2022:</strong> <span class="positive-value">Normal Study</span></p>
        </div>
        <div class="metric-container" style="margin-top: 15px;">
            <div class="metric-item">
                <div class="metric-value highlight-value">4</div>
                <div class="metric-label">Total Scans</div>
            </div>
            <div class="metric-item">
                <div class="metric-value highlight-value">1</div>
                <div class="metric-label">Positive Findings</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Tabs for detailed analysis ---
st.markdown("<h3>Detailed Analysis</h3>", unsafe_allow_html=True)
tabs = st.tabs(["Flow Analysis", "Compression Test", "Vessel Comparison", "AI Detection"])

with tabs[0]:
    flow_col1, flow_col2 = st.columns([2, 1])
    
    with flow_col1:
        st.caption("Spectral Doppler Waveform Analysis")
        
        # Create subplot with shared x-axis
        fig_doppler = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   specs=[[{"type": "scatter"}], [{"type": "scatter"}]],
                                   subplot_titles=("Normal Vein - Phasic Flow", "DVT Affected Vein - Diminished Flow"),
                                   vertical_spacing=0.12)
        
        # Time points
        x_time = np.linspace(0, 4, 200)
        
        # Normal vein data
        normal_data = 20 + 12 * np.sin(2*np.pi*x_time) + 5 * np.sin(6*np.pi*x_time)
        
        # Abnormal DVT data
        dvt_data = 10 + 4 * np.sin(2*np.pi*x_time) + np.random.normal(0, 1, 200)
        
        fig_doppler.add_trace(
            go.Scatter(x=x_time, y=normal_data, mode='lines', name='Normal Vein',
                      line=dict(color='rgba(0,128,96,0.8)', width=2)),
            row=1, col=1
        )
        
        fig_doppler.add_trace(
            go.Scatter(x=x_time, y=dvt_data, mode='lines', name='DVT Affected Vein',
                      line=dict(color='rgba(128,0,32,0.8)', width=2)),
            row=2, col=1
        )
        
        # Add annotations highlighting key differences
        fig_doppler.add_annotation(
            x=1.5, y=30, text="Normal phasic flow pattern",
            showarrow=True, arrowhead=2, ax=20, ay=-30, row=1, col=1,
            font=dict(color="#008060", size=12)
        )
        
        fig_doppler.add_annotation(
            x=2.5, y=12, text="Diminished flow amplitude",
            showarrow=True, arrowhead=2, ax=-40, ay=-20, row=2, col=1,
            font=dict(color="#800020", size=12)
        )
        
        fig_doppler.update_layout(
            height=500, 
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor="rgba(240,247,255,0.5)"
        )
        
        fig_doppler.update_xaxes(title_text="Time (s)", row=2, col=1, gridcolor="#e6f0ff")
        fig_doppler.update_yaxes(title_text="Flow Velocity (cm/s)", row=1, col=1, gridcolor="#e6f0ff")
        fig_doppler.update_yaxes(title_text="Flow Velocity (cm/s)", row=2, col=1, gridcolor="#e6f0ff")
        
        flow_chart = st.plotly_chart(fig_doppler, use_container_width=True)
    
    with flow_col2:
        st.subheader("Flow Analysis Metrics")
        
        # Modern styled table instead of pandas DataFrame
        st.markdown("""
        <div style="background-color: #f0f7ff; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
            <table style="width:100%; border-collapse: separate; border-spacing: 0;">
                <thead>
                    <tr>
                        <th style="padding: 12px; text-align: left; background-color: #d0e0ff; color: #0a2540; font-weight: 600;">Metric</th>
                        <th style="padding: 12px; text-align: center; background-color: #d0e0ff; color: #0a2540; font-weight: 600;">Value</th>
                        <th style="padding: 12px; text-align: center; background-color: #d0e0ff; color: #0a2540; font-weight: 600;">Normal Range</th>
                        <th style="padding: 12px; text-align: center; background-color: #d0e0ff; color: #0a2540; font-weight: 600;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff;">Peak Systolic Velocity</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff; text-align: center; font-weight: 500;">8.4 cm/s</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff; text-align: center;">15-25 cm/s</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff; text-align: center; color: #800020;">Abnormal ⚠️</td>
                    </tr>
                    <tr style="background-color: rgba(255,255,255,0.5);">
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff;">End Diastolic Velocity</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff; text-align: center; font-weight: 500;">4.2 cm/s</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff; text-align: center;">5-10 cm/s</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff; text-align: center; color: #800020;">Abnormal ⚠️</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff;">Resistance Index</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff; text-align: center; font-weight: 500;">0.5</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff; text-align: center;">0.8-1.0</td>
                        <td style="padding: 10px; border-bottom: 1px solid #e6f0ff; text-align: center; color: #800020;">Abnormal ⚠️</td>
                    </tr>
                    <tr style="background-color: rgba(255,255,255,0.5);">
                        <td style="padding: 10px;">Augmentation</td>
                        <td style="padding: 10px; text-align: center; font-weight: 500;">Absent</td>
                        <td style="padding: 10px; text-align: center;">Present</td>
                        <td style="padding: 10px; text-align: center; color: #800020;">Abnormal ⚠️</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Flow Analysis Interpretation:**
        
        The spectral waveform demonstrates significantly reduced velocity and loss of normal phasicity, consistent with venous obstruction. 
        
        Normal venous flow shows respiratory variation with augmentation during respiratory maneuvers, which is absent in the affected vessel.
        
        These findings strongly suggest presence of a thrombus impeding normal blood flow.
        """)

with tabs[1]:
    comp_col1, comp_col2 = st.columns([2, 1])
    
    with comp_col1:
        st.caption("Vein Compression Test Analysis")
        
        # Create data for normal vs affected vein compression
        compression_levels = ['No compression', '25% compression', '50% compression', '75% compression', '100% compression']
        normal_vein_area = [100, 75, 50, 25, 0]  # Normal vein should fully compress
        dvt_vein_area = [100, 95, 90, 85, 80]    # DVT vein doesn't compress fully
        
        fig_compression = go.Figure()
        
        # Add bars for normal vein
        fig_compression.add_trace(go.Bar(
            x=compression_levels,
            y=normal_vein_area,
            name='Normal Vein',
            marker_color='rgba(0,128,96,0.7)'
        ))
        
        # Add bars for DVT affected vein
        fig_compression.add_trace(go.Bar(
            x=compression_levels,
            y=dvt_vein_area,
            name='DVT Affected Vein',
            marker_color='rgba(128,0,32,0.7)'
        ))
        
        # Add annotations
        fig_compression.add_annotation(
            x='100% compression', 
            y=5, 
            text="Complete compression (normal)",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color="#008060", size=12)
        )
        
        fig_compression.add_annotation(
            x='100% compression', 
            y=85, 
            text="Incompressible vein (DVT)",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30,
            font=dict(color="#800020", size=12)
        )
        
        fig_compression.update_layout(
            title='Vein Compression Test Comparison',
            xaxis_title='Applied Compression',
            yaxis_title='Remaining Vessel Area (%)',
            barmode='group',
            template='plotly_white',
            height=450,
            plot_bgcolor="rgba(240,247,255,0.5)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )
        
        st.plotly_chart(fig_compression, use_container_width=True)
    
    with comp_col2:
        st.subheader("Compression Test Results")
        
        st.markdown("""
        <div class="info-card">
            <div class="card-header">Normal Vein Behavior</div>
            <p>Complete collapse of vessel lumen with gentle pressure</p>
            <p>• <span class="positive-value">100%</span> compressible</p>
            <p>• Normal wall apposition</p>
            <p>• Rebounds to normal size when pressure released</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div class="card-header">DVT Affected Vein</div>
            <p>Resistant to compression due to thrombus</p>
            <p>• Only <span class="highlight-value">20%</span> compressible</p>
            <p>• Firm resistance to probe pressure</p>
            <p>• Visible thrombus material within lumen</p>
            <p>• <span class="highlight-value">Non-compressible segment length: 2.5 cm</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Interpretation:**
        
        Lack of vein compressibility is the gold standard ultrasound finding for DVT diagnosis.
        
        This patient's femoral vein demonstrates significant resistance to compression,
        strongly indicating the presence of thrombus within the vessel lumen.
        """)

with tabs[2]:
    vessel_col1, vessel_col2 = st.columns([3, 1])
    
    with vessel_col1:
        st.caption("Vessel Diameter and Flow Comparison")
        
        # Create data for vessel comparison
        vessel_locations = ['Proximal', 'Mid-segment', 'Distal', 'Contralateral']
        
        # Vessel diameter data
        diameter_normal = [7.5, 7.2, 6.8, 7.4]
        diameter_affected = [10.2, 9.8, 8.2, 7.5]
        
        # Create subplot with shared x-axis
        fig_vessel = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                  subplot_titles=("Vessel Diameter (mm)", "Flow Velocity (cm/s)"),
                                  vertical_spacing=0.15)
        
        # Add diameter traces
        fig_vessel.add_trace(
            go.Scatter(
                x=vessel_locations, 
                y=diameter_affected,
                mode='lines+markers',
                name='DVT Affected Vein',
                line=dict(color='rgba(128,0,32,0.8)', width=3),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        fig_vessel.add_trace(
            go.Scatter(
                x=vessel_locations, 
                y=diameter_normal,
                mode='lines+markers',
                name='Normal Range',
                line=dict(color='rgba(0,128,96,0.8)', width=3),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Flow velocity data
        velocity_normal = [19.3, 18.5, 17.2, 19.0]
        velocity_affected = [8.4, 7.2, 10.5, 18.8]
        
        # Add velocity traces
        fig_vessel.add_trace(
            go.Scatter(
                x=vessel_locations, 
                y=velocity_affected,
                mode='lines+markers',
                name='DVT Affected Vein',
                line=dict(color='rgba(128,0,32,0.8)', width=3),
                marker=dict(size=10),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig_vessel.add_trace(
            go.Scatter(
                x=vessel_locations, 
                y=velocity_normal,
                mode='lines+markers',
                name='Normal Range',
                line=dict(color='rgba(0,128,96,0.8)', width=3),
                marker=dict(size=10),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add annotations
        fig_vessel.add_annotation(
            x='Proximal', 
            y=10.7, 
            text="Enlarged vessel due to thrombus",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30,
            row=1, col=1,
            font=dict(color="#800020", size=12)
        )
        
        fig_vessel.add_annotation(
            x='Mid-segment', 
            y=6.2, 
            text="Reduced flow velocity",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=30,
            row=2, col=1,
            font=dict(color="#800020", size=12)
        )
        
        fig_vessel.update_layout(
            height=500,
            template='plotly_white',
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor="rgba(240,247,255,0.5)"
        )
        
        st.plotly_chart(fig_vessel, use_container_width=True)
    
    with vessel_col2:
        st.subheader("Vessel Metrics")
        
        st.markdown("""
        <div class="info-card">
            <div class="card-header">Normal vs. Affected</div>
            <table style="width:100%">
                <tr>
                    <td><strong>Normal diameter:</strong></td>
                    <td class="positive-value">7.5 mm</td>
                </tr>
                <tr>
                    <td><strong>Affected diameter:</strong></td>
                    <td class="highlight-value">10.2 mm</td>
                </tr>
                <tr>
                    <td><strong>Diameter increase:</strong></td>
                    <td class="highlight-value">+36%</td>
                </tr>
                <tr>
                    <td><strong>Flow reduction:</strong></td>
                    <td class="highlight-value">-56%</td>
                </tr>
                <tr>
                    <td><strong>Wall thickness:</strong></td>
                    <td class="highlight-value">Increased</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Vessel Analysis Interpretation:**
        
        The affected vein shows significant dilation compared to normal values and the contralateral vessel.
        
        Flow velocity is markedly reduced in the affected segments, with normalization only in the distal, uninvolved segment.
        
        The pattern of vessel enlargement with reduced flow is classic for partial venous occlusion by thrombus.
        """)

with tabs[3]:
    ai_col1, ai_col2 = st.columns([2, 1])
    
    with ai_col1:
        st.caption("AI Detection Heatmap and Confidence Metrics")
        
        # Create a figure for the AI detection confidence
        fig_ai = go.Figure()
        
        # Frame numbers
        frames = list(range(1, 101))
        
        # AI detection confidence over time (simulated data)
        confidence_scores = [
            60 + 20 * np.sin(i/10) for i in range(30)  # Initial scanning
        ] + [
            85 + 10 * np.sin(i/5) for i in range(40)   # DVT detection period
        ] + [
            90 + 5 * np.sin(i/8) for i in range(30)    # Continued high confidence
        ]
        
        # Add the confidence trace
        fig_ai.add_trace(go.Scatter(
            x=frames,
            y=confidence_scores,
            mode='lines',
            name='DVT Detection Confidence',
            line=dict(
                color='rgba(128,0,32,0.9)',
                width=3,
                shape='spline'
            ),
            fill='tozeroy',
            fillcolor='rgba(128,0,32,0.1)'
        ))
        
        # Add a horizontal line at 75% confidence threshold
        fig_ai.add_shape(
            type="line",
            x0=1,
            y0=75,
            x1=100,
            y1=75,
            line=dict(
                color="rgba(0,0,0,0.5)",
                width=2,
                dash="dash",
            )
        )
        
        fig_ai.add_annotation(
            x=20,
            y=75,
            text="Detection Threshold (75%)",
            showarrow=False,
            yshift=10,
            font=dict(size=12)
        )
        
        # Add regions
        fig_ai.add_vrect(
            x0=30, x1=70,
            fillcolor="rgba(128,0,32,0.1)",
            opacity=0.5,
            layer="below",
            line_width=0
        )
        
        fig_ai.add_annotation(
            x=50,
            y=95,
            text="DVT Detected",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30,
            font=dict(color="#800020", size=14, family="Arial Black")
        )
        
        # Update layout
        fig_ai.update_layout(
            title='AI DVT Detection Confidence Over Time',
            xaxis_title='Frame Number',
            yaxis_title='Confidence Score (%)',
            template='plotly_white',
            height=450,
            plot_bgcolor="rgba(240,247,255,0.5)",
            yaxis=dict(range=[0, 100]),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_ai, use_container_width=True)
    
    with ai_col2:
        st.subheader("AI Detection Results")
        
        st.markdown("""
        <div class="info-card">
            <div class="card-header">Detection Metrics</div>
            <div class="metric-container">
                <div class="metric-item">
                    <div class="metric-value highlight-value">95%</div>
                    <div class="metric-label">Peak Confidence</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value highlight-value">87%</div>
                    <div class="metric-label">Average Confidence</div>
                </div>
            </div>
            <div class="metric-container" style="margin-top: 10px;">
                <div class="metric-item">
                    <div class="metric-value highlight-value">41/100</div>
                    <div class="metric-label">Frames with Detection</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">12ms</div>
                    <div class="metric-label">Processing Time/Frame</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <div class="card-header">Key AI Findings</div>
            <ul>
                <li><span class="highlight-value">Proximal femoral vein</span> - High confidence detection</li>
                <li>Non-compressible segment length: 2.5 cm</li>
                <li>Increased vessel diameter: +36%</li>
                <li>Reduced flow velocity: -56%</li>
                <li>Confidence score exceeds clinical threshold (75%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **AI Interpretation:**
        
        The AI detection system has identified a deep vein thrombosis with high confidence (95%).
        
        The detection was consistent across multiple frames and analysis methods.
        
        This detection meets the clinical threshold for recommended follow-up.
        """)
