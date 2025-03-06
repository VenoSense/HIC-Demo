import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager
import shutil
import traceback

# --- Page Configuration ---
st.set_page_config(layout="wide")

# --- Check for query parameters to access hidden page ---
query_params = st.query_params
page = query_params.get("page", ["main"])[0]

# --- Custom CSS for styling with new color scheme ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: white;
    }
    
    /* Card styling */
    .info-card {
        background-color: #f0f7ff;  /* Light blue background */
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 3px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border: 1px solid #e6f0ff;
    }
    
    /* Card header styling */
    .card-header {
        color: #102040;  /* Dark blue text */
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 10px;
        border-bottom: 1px solid #d0e0ff;
        padding-bottom: 5px;
    }
    
    /* Alert styling */
    .alert-danger {
        background-color: rgba(128,0,32,0.08);  /* Maroon background */
        color: #800020;  /* Maroon text */
        border-left: 5px solid #800020;
        padding: 10px;
        margin: 10px 0;
        border-radius: 3px;
    }
    
    /* Highlight color for important metrics */
    .highlight-value {
        color: #800020;  /* Maroon */
        font-weight: bold;
    }
    
    /* Secondary highlight for positive values */
    .positive-value {
        color: #008060;  /* Teal green */
        font-weight: bold;
    }
    
    /* Metrics styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    
    .metric-item {
        flex: 1 1 48%;
        margin: 5px;
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e6f0ff;
    }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #102040;  /* Dark blue */
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #6080a0;  /* Medium blue */
    }
    
    /* Logo container */
    .logo-container {
        position: absolute;
        top: 10px;
        left: 10px;
        background-color: white;
        padding: 5px;
        border-radius: 5px;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
        font-weight: 400;
        color: #102040;  /* Dark blue */
    }

    .stTabs [aria-selected="true"] {
        background-color: #f0f7ff;  /* Light blue */
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
        background-color: #800020;  /* Maroon */
    }
    
    .status-warning {
        background-color: #FFD700;  /* Yellow */
    }
    
    .status-normal {
        background-color: #008060;  /* Teal green */
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #102040;  /* Dark blue */
    }
    
    /* Speed control pills styling */
    .speed-pills {
        background-color: #f0f7ff;
        border-radius: 20px;
        overflow: hidden;
        display: inline-flex;
        margin: 10px 0;
        border: 1px solid #d0e0ff;
    }
    
    .speed-option {
        padding: 5px 10px;
        cursor: pointer;
        font-size: 0.8rem;
        background-color: transparent;
        border: none;
        color: #102040;
    }
    
    .speed-option.active {
        background-color: #102040;
        color: white;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Page title and spacing adjustments */
    h1, h2, h3, h4, h5, h6 {
        color: #102040;  /* Dark blue */
    }
    
    /* Fix heading margins */
    .block-container {
        padding-top: 30px;
    }
    
    /* Video selector styling */
    .video-selector {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Yellow accent for some elements */
    .accent-yellow {
        color: #FFD700 !important;
    }
    
    /* Footer styling */
    footer {
        visibility: hidden;
    }
    
    /* Hide hamburger menu */
    #MainMenu {
        visibility: hidden;
    }
    
    /* Video controls container */
    .video-controls {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        margin-top: 5px;
        border: 1px solid #e6f0ff;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper functions ---
def ensure_video_path(video_path):
    """
    Ensure the video path exists and is accessible.
    If not, try to find an alternative or return a default.
    """
    # First check if the exact path exists
    if os.path.isfile(video_path) and os.access(video_path, os.R_OK):
        return video_path
    
    # If path doesn't exist, check if it's just a filename and prepend videos/ directory
    if not video_path.startswith("videos/"):
        potential_path = os.path.join("videos", os.path.basename(video_path))
        if os.path.isfile(potential_path) and os.access(potential_path, os.R_OK):
            return potential_path
    
    # Check if we have any videos in the directory as fallback
    video_dir = "videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)
    
    available_videos = [str(f) for f in Path(video_dir).glob("*.mp4")]
    if available_videos:
        st.warning(f"Could not find {video_path}. Using {available_videos[0]} instead.")
        return available_videos[0]
    
    # If no videos found, return a path to an error image or create a blank frame
    st.error(f"No video files found in {video_dir}. Please add some video files.")
    return None

def get_available_videos():
    """Get a list of all available video files in the videos directory."""
    video_dir = "videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)
        return []
    
    return [f.name for f in Path(video_dir).glob("*.mp4")]

def debug_video_access(file_path):
    """Debug video file access issues and provide detailed information."""
    debug_info = {
        "exists": os.path.exists(file_path),
        "is_file": os.path.isfile(file_path),
        "size": os.path.getsize(file_path) if os.path.exists(file_path) else "N/A",
        "absolute_path": os.path.abspath(file_path),
        "readable": os.access(file_path, os.R_OK) if os.path.exists(file_path) else False,
    }
    
    # Try to open with opencv
    cap = cv2.VideoCapture(file_path)
    debug_info["opencv_opened"] = cap.isOpened()
    if cap.isOpened():
        debug_info["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        debug_info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        debug_info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        debug_info["fps"] = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    
    return debug_info

@contextmanager
def safe_video_capture(file_path):
    """
    A context manager that safely creates a video capture object,
    making a temporary copy of the video if necessary.
    """
    # First try to open directly - this is the most efficient
    cap = cv2.VideoCapture(file_path)
    temp_dir = None
    temp_file = None
    
    try:
        # If direct opening failed, try with a temporary copy
        if not cap.isOpened():
            temp_dir = tempfile.mkdtemp()
            video_name = os.path.basename(file_path)
            temp_file = os.path.join(temp_dir, video_name)
            
            # Copy the file to a temporary location
            st.info(f"Creating temporary copy of video to resolve access issues...")
            shutil.copy2(file_path, temp_file)
            
            # Close the original capture and try with the temp file
            cap.release()
            cap = cv2.VideoCapture(temp_file)
            
            if not cap.isOpened():
                st.error(f"Failed to open video even with temporary copy.")
                debug_info = debug_video_access(file_path)
                st.json(debug_info)
                
                # Try one last approach - memory mapping
                try:
                    with open(file_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    # Create a temporary file and write the bytes
                    temp_file2 = os.path.join(temp_dir, "streamed_" + video_name)
                    with open(temp_file2, 'wb') as f:
                        f.write(video_bytes)
                    
                    cap = cv2.VideoCapture(temp_file2)
                except Exception as e:
                    st.error(f"Memory mapping approach failed too: {e}")
        
        yield cap
        
    finally:
        # Make sure we clean up resources
        cap.release()
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        return YOLO("weights/pigv3.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'weights/pigv3.pt' exists.")
        return None

model = load_model()
if model is None:
    st.stop()

# ====================== HEATMAP VIDEO GENERATOR PAGE ======================
if page == "generate":
    st.title("Heatmap Video Generator")
    st.markdown("This hidden page allows you to generate pre-processed videos with heatmaps that can be used in the main interface.")
    
    # Input parameters
    st.header("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_video = st.selectbox(
            "Select Input Video", 
            ["videos/demo.mp4", "videos/prerender.mp4"] + 
            [str(f) for f in Path("videos").glob("*.mp4") if f.name not in ["demo.mp4", "prerender.mp4"]]
        )
        
        output_filename = st.text_input("Output Filename", "prerender.mp4")
        
        if not output_filename.endswith('.mp4'):
            output_filename += '.mp4'
            
        output_path = os.path.join("videos", output_filename)
        
        # Check if output would overwrite existing file
        if os.path.exists(output_path):
            st.warning(f"Warning: {output_path} already exists and will be overwritten.")
    
    with col2:
        colormap = st.selectbox(
            "Heatmap Style", 
            ["COLORMAP_JET", "COLORMAP_HOT", "COLORMAP_INFERNO", "COLORMAP_VIRIDIS", "COLORMAP_PLASMA"],
            index=0
        )
        
        heatmap_alpha = st.slider("Heatmap Intensity", 0.1, 1.0, 0.6, 0.1)
        heatmap_decay = st.slider("Heatmap Persistence", 0.8, 0.99, 0.95, 0.01,
                                 help="Higher values make heatmap effects persist longer")
        
    # Options
    include_boxes = st.checkbox("Include bounding boxes in output", value=True)
    include_confidence = st.checkbox("Show confidence scores", value=True)
    
    # Generate button
    generate_button = st.button("Generate Heatmap Video", use_container_width=True)
    
    if generate_button:
        # Open source video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            st.error(f"Error opening video: {input_video}")
            st.stop()
            
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Display a preview area
        preview_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize heatmap accumulator
        heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Process the video
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Run inference on the frame
                results = model(frame, stream=True)
                
                # Process detection results
                detections_this_frame = np.zeros_like(heatmap_accumulator)
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Draw bounding boxes if enabled
                        if include_boxes:
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Add confidence text if enabled
                            if include_confidence:
                                text = f"{confidence:.2f}"
                                cv2.putText(display_frame, text, (x1, y1-5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Create heatmap intensity based on detection confidence
                        intensity = confidence * 255
                        
                        # Add a Gaussian blob centered on the detection
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Size the Gaussian based on the detection size
                        sigma_x = max(2, (x2 - x1) // 4)
                        sigma_y = max(2, (y2 - y1) // 4)
                        
                        # Create a mask for the current detection
                        y_coords, x_coords = np.ogrid[:detections_this_frame.shape[0], :detections_this_frame.shape[1]]
                        
                        # Calculate distance from center (vectorized)
                        dist_from_center = ((x_coords - center_x) ** 2) / (2 * sigma_x ** 2) + \
                                           ((y_coords - center_y) ** 2) / (2 * sigma_y ** 2)
                        
                        # Create Gaussian mask
                        mask = intensity * np.exp(-dist_from_center)
                        
                        # Update the detection frame
                        detections_this_frame = np.maximum(detections_this_frame, mask)
                
                # Apply decay factor to existing heatmap
                heatmap_accumulator *= heatmap_decay
                
                # Add new detections
                heatmap_accumulator = np.maximum(heatmap_accumulator, detections_this_frame)
                
                # Apply heatmap to display frame
                if np.max(heatmap_accumulator) > 0:
                    normalized_heatmap = cv2.normalize(
                        heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)
                    
                    # Apply colormap
                    colored_heatmap = cv2.applyColorMap(normalized_heatmap, getattr(cv2, colormap))
                    
                    # Blend with original frame
                    alpha = heatmap_alpha
                    beta = 1 - alpha
                    display_frame = cv2.addWeighted(display_frame, beta, colored_heatmap, alpha, 0)
                
                # Write frame to output video
                out.write(display_frame)
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
                
                # Show preview every 10 frames to avoid slowing down processing
                if frame_count % 10 == 0:
                    preview_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), 
                                            caption=f"Preview (frame {frame_count})")
            
            # Release resources
            cap.release()
            out.release()
            
            status_text.text(f"✅ Processing complete! Video saved to {output_path}")
            
            # Show download button
            with open(output_path, 'rb') as file:
                st.download_button(
                    label="Download processed video",
                    data=file,
                    file_name=output_filename,
                    mime="video/mp4"
                )
                
            # Add button to return to main page
            if st.button("Return to Main Interface"):
                st.experimental_set_query_params(page="main")
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            # Try to clean up resources
            try:
                cap.release()
                out.release()
            except:
                pass
    
    # Add a note at the bottom
    st.markdown("---")
    st.markdown("To access this page, add `?page=generate` to the URL. To return to the main page, remove it or use `?page=main`.")
            
# ====================== MAIN APPLICATION PAGE ======================
else:  # Default page is main
    # --- Display Logo in corner ---
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image("assets/logo.png", width=80)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add some spacing for the logo
    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)

    # --- Settings Menu ---
    with st.sidebar.expander("Settings", expanded=True):
        demo_mode = st.radio("Demo Mode", ["Live", "Prerendered"])
        
        # Get available videos from the videos directory
        available_videos = get_available_videos()
        
        if len(available_videos) == 0:
            st.error("No video files found in the videos directory. Please add some .mp4 files.")
            # Create a placeholder entry to avoid errors
            available_videos = ["no_videos_available.mp4"]
        
        if demo_mode == "Live":
            inference_speed = st.slider("Inference Speed", 0.1, 1.0, 0.5, 0.1)
            colormap_options = ["COLORMAP_JET", "COLORMAP_HOT", "COLORMAP_INFERNO", "COLORMAP_VIRIDIS", "COLORMAP_PLASMA"]
            selected_colormap = st.selectbox("Heatmap Style", colormap_options, index=0)
            heatmap_intensity = st.slider("Heatmap Intensity", 0.1, 1.0, 0.6, 0.1)
            
            # Default to demo.mp4 if available, otherwise use the first available video
            default_idx = available_videos.index("demo.mp4") if "demo.mp4" in available_videos else 0
            selected_video = st.selectbox("Select Video", available_videos, index=default_idx)
            video_source = os.path.join("videos", selected_video)
        else:  # Prerendered mode
            # Initialize playback speed if not already set
            if "playback_speed" not in st.session_state:
                st.session_state.playback_speed = 1.0
            
            # Default to prerender.mp4 if available, otherwise use the first available video    
            default_idx = available_videos.index("prerender.mp4") if "prerender.mp4" in available_videos else 0
            selected_video = st.selectbox("Select Video", available_videos, index=default_idx)
            video_source = os.path.join("videos", selected_video)
            
            # Playback speed control in sidebar for prerendered videos
            st.subheader("Playback Speed")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # Create speed buttons in columns
            speed_cols = [col1, col2, col3, col4, col5]
            speeds = [1.0, 2.0, 3.0, 4.0, 5.0]
            
            # Draw speed selector buttons
            for i, (col, speed) in enumerate(zip(speed_cols, speeds)):
                with col:
                    if st.button(f"{int(speed)}x", key=f"speed_btn_{speed}", 
                               use_container_width=True,
                               type="primary" if st.session_state.playback_speed == speed else "secondary"):
                        st.session_state.playback_speed = speed

    # --- Ensure video path is valid and prepare video source ---
    try:
        # Ensure videos directory exists
        if not os.path.exists("videos"):
            os.makedirs("videos", exist_ok=True)
        
        # Define page layout first (moved from below)
        # Top status bar with current date and patient info
        current_date = datetime.now().strftime("%m/%d/%Y")

        col_date, col_patient, col_scan = st.columns(3)
        with col_date:
            st.markdown(f"<div class='info-card'><div class='card-header'>Date</div>{current_date}</div>", unsafe_allow_html=True)
        with col_patient:
            st.markdown("<div class='info-card'><div class='card-header'>Patient</div>John Doe (ID: 12345678)</div>", unsafe_allow_html=True)
        with col_scan:
            st.markdown("<div class='info-card'><div class='card-header'>Previous Scan</div>02/05/2025 - No clot detected</div>", unsafe_allow_html=True)

        # Main content area with central ultrasound and surrounding blocks
        # Create a 3x3 grid layout
        row1_col1, row1_col2, row1_col3 = st.columns([1, 2, 1])
        row2_col1, row2_col2, row2_col3 = st.columns([1, 2, 1])
        row3_col1, row3_col2, row3_col3 = st.columns([1, 2, 1])

        # ... include all the layout code from before...

        # IMPORTANT: Define the video_placeholder here, before using it later
        with row2_col2:
            st.markdown("<h3 style='text-align: center;'>Ultrasound with Detection</h3>", unsafe_allow_html=True)
            video_placeholder = st.empty()  # Placeholder for the video stream
            
            # Controls under the video - only show heatmap controls for Live mode
            if demo_mode == "Live":
                ctrl_col1, ctrl_col2 = st.columns(2)
                with ctrl_col1:
                    heatmap_toggle = st.checkbox("Show Heatmap", value=st.session_state.heatmap_activated)
                    st.session_state.heatmap_activated = heatmap_toggle
                with ctrl_col2:
                    reset_button = st.button("Reset Heatmap")
                    if reset_button:
                        st.session_state.heatmap_accumulator = np.zeros_like(st.session_state.heatmap_accumulator)
                        st.session_state.frame_count = 0

        # ... Rest of the layout code...

        # Check if selected video exists
        video_exists = os.path.isfile(video_source)
        if not video_exists:
            st.warning(f"Video file not found: {video_source}")
            
            # Look for available videos
            available_videos = get_available_videos()
            if available_videos:
                video_source = os.path.join("videos", available_videos[0])
                st.info(f"Using alternative video: {video_source}")
            else:
                # Create a placeholder video if none exist
                st.error("No video files available. Creating a placeholder video...")
                placeholder_path = os.path.join("videos", "placeholder.mp4")
                
                # Create a blank video
                height, width = 480, 640
                fps = 30
                seconds = 5
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(placeholder_path, fourcc, fps, (width, height))
                
                # Create and write frames
                for i in range(fps * seconds):
                    # Create a blank frame with frame number
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(frame, f"Placeholder Frame {i+1}", (50, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    out.write(frame)
                
                out.release()
                video_source = placeholder_path
        
        # Debug video access if needed
        # debug_info = debug_video_access(video_source)
        # st.write("Video Debug Info:", debug_info)
        
        # Initialize session states and video source
        if 'video_source' not in st.session_state:
            st.session_state.video_source = video_source
        elif st.session_state.video_source != video_source:
            # Source has changed, reset states
            if 'cap' in st.session_state:
                st.session_state.cap.release()
            if 'heatmap_accumulator' in st.session_state:
                del st.session_state['heatmap_accumulator']
            if 'frame_count' in st.session_state:
                st.session_state.frame_count = 0
            st.session_state.video_source = video_source
        
        # Now that we have the video placeholder defined, we can use it
        # Use our safe video capture context manager
        with safe_video_capture(video_source) as cap:
            if cap is None or not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_source}")
            
            # Get video properties for initializing the application
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            # Initialize session states needed for video processing
            if 'heatmap_accumulator' not in st.session_state:
                st.session_state.heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)
            
            if 'frame_count' not in st.session_state:
                st.session_state.frame_count = 0
            
            if 'heatmap_activated' not in st.session_state:
                st.session_state.heatmap_activated = True
            
            # Read a test frame to verify we can access the video content
            ret, test_frame = cap.read()
            if not ret:
                raise ValueError("Could not read frames from the video")
            
            # Reset position to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Store the validated source
            st.session_state.valid_video_source = video_source
            
            # Process and display video frames
            def process_frame():
                # Create a fresh video capture for each processing cycle
                with safe_video_capture(st.session_state.valid_video_source) as frame_cap:
                    if not frame_cap.isOpened():
                        return None
                    
                    # Set position based on frame count if needed
                    if st.session_state.frame_count > 0:
                        # Modulo by total frames to loop the video
                        total_frames = int(frame_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if total_frames > 0:  # Avoid division by zero
                            target_frame = st.session_state.frame_count % total_frames
                            frame_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    
                    # Read the frame
                    ret, frame = frame_cap.read()
                    if not ret:
                        # Try reading from beginning if at end
                        frame_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = frame_cap.read()
                        if not ret:
                            return None
                    
                    # Increment frame count
                    st.session_state.frame_count += 1
                    
                    # Rest of frame processing logic
                    # ...existing frame processing code...
                    
                    # Return processed frame
                    return processed_frame
            
            # Run the video processing loop
            try:
                last_update_time = time.time()
                while True:
                    # Calculate time to sleep for frame rate control
                    current_time = time.time()
                    elapsed = current_time - last_update_time
                    
                    # Control playback speed
                    target_frame_time = 1.0 / (30.0 * st.session_state.get('playback_speed', 1.0))
                    if demo_mode != "Live" and elapsed < target_frame_time:
                        time.sleep(target_frame_time - elapsed)
                    
                    # Get and process a new frame
                    processed_frame = process_frame()
                    last_update_time = time.time()
                    
                    if processed_frame is not None:
                        # Update the video display
                        video_placeholder.image(processed_frame, channels="RGB")
                    else:
                        st.warning("End of video or processing error.")
                        time.sleep(1)  # Avoid excessive CPU usage on errors
                        break
            except Exception as e:
                st.error(f"Video playback error: {e}")
                st.code(traceback.format_exc())
                
    except Exception as e:
        st.error(f"Failed to initialize video: {e}")
        st.code(traceback.format_exc())
        
        # Display a blank placeholder
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Video Unavailable", (180, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if video_placeholder:
            video_placeholder.image(blank_frame, channels="BGR")

    # --- Initialize session states and video source ---
    if 'video_source' not in st.session_state:
        st.session_state.video_source = video_source
    elif st.session_state.video_source != video_source:
        # Source has changed, reset states
        if 'cap' in st.session_state:
            st.session_state.cap.release()
        if 'heatmap_accumulator' in st.session_state:
            del st.session_state['heatmap_accumulator']
        if 'frame_count' in st.session_state:
            st.session_state.frame_count = 0
        st.session_state.video_source = video_source
    
    # --- Prepare video capture ---
    def get_cap(source):
        cap = cv2.VideoCapture(source)
        # Double check if capture is successfully opened
        if not cap.isOpened():
            st.error(f"Failed to open video: {source}")
            return None
        return cap

    if 'cap' not in st.session_state or st.session_state.video_source != video_source:
        st.session_state.cap = get_cap(video_source)
        
    cap = st.session_state.cap
    
    if cap is None or not cap.isOpened():
        st.error(f"Error opening video stream from {video_source}. The file may be corrupted or in use by another process.")
        
        # Create a blank frame as placeholder
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Video Not Available", (180, 240), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        st.image(blank_frame, channels="BGR")
        st.stop()

    # --- Initialize session state ---
    try:
        if 'processed_frame' not in st.session_state:
            ret, initial_frame = cap.read()
            if ret:
                st.session_state.processed_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB)
            else:
                st.error("Could not read the first frame from the video source.")
                st.stop()

        if 'heatmap_accumulator' not in st.session_state:
            st.session_state.heatmap_accumulator = np.zeros((
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ), dtype=np.float32)

        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0

        if 'heatmap_activated' not in st.session_state:
            st.session_state.heatmap_activated = True
    except Exception as e:
        st.error(f"Error initializing video: {e}")
        st.stop()

    # --- Page Layout ---
    # Top status bar with current date and patient info
    current_date = datetime.now().strftime("%m/%d/%Y")

    col_date, col_patient, col_scan = st.columns(3)
    with col_date:
        st.markdown(f"<div class='info-card'><div class='card-header'>Date</div>{current_date}</div>", unsafe_allow_html=True)
    with col_patient:
        st.markdown("<div class='info-card'><div class='card-header'>Patient</div>John Doe (ID: 12345678)</div>", unsafe_allow_html=True)
    with col_scan:
        st.markdown("<div class='info-card'><div class='card-header'>Previous Scan</div>02/05/2025 - No clot detected</div>", unsafe_allow_html=True)

    # Main content area with central ultrasound and surrounding blocks
    # Create a 3x3 grid layout
    row1_col1, row1_col2, row1_col3 = st.columns([1, 2, 1])
    row2_col1, row2_col2, row2_col3 = st.columns([1, 2, 1])
    row3_col1, row3_col2, row3_col3 = st.columns([1, 2, 1])

    # Top Row - Alert and Findings
    with row1_col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-header">Confidence</div>
            <div class="metric-container">
                <div class="metric-item">
                    <div class="metric-value highlight-value">95%</div>
                    <div class="metric-label">Model Confidence</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">High</div>
                    <div class="metric-label">Certainty</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with row1_col2:
        st.markdown("""
        <div class="alert-danger">
            <h3><span class="status-indicator status-critical"></span> DVT ALERT: Clot Detected</h3>
            <p>Urgent evaluation required</p>
        </div>
        """, unsafe_allow_html=True)

    with row1_col3:
        st.markdown("""
        <div class="info-card">
            <div class="card-header">Key Findings</div>
            <ul>
                <li><strong>Location:</strong> <span class="highlight-value">Proximal femoral vein</span></li>
                <li><strong>Size:</strong> 2.5 CM</li>
                <li><strong>Flow Disruption:</strong> Partial</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Middle Row - Ultrasound Video
    with row2_col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-header">Flow Metrics</div>
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
        </div>
        """, unsafe_allow_html=True)

    # Center column - Main ultrasound display
    with row2_col2:
        st.markdown("<h3 style='text-align: center;'>Ultrasound with Detection</h3>", unsafe_allow_html=True)
        video_placeholder = st.empty()  # Placeholder for the video stream
        
        # Controls under the video - only show heatmap controls for Live mode
        if demo_mode == "Live":
            ctrl_col1, ctrl_col2 = st.columns(2)
            with ctrl_col1:
                heatmap_toggle = st.checkbox("Show Heatmap", value=st.session_state.heatmap_activated)
                st.session_state.heatmap_activated = heatmap_toggle
            with ctrl_col2:
                reset_button = st.button("Reset Heatmap")
                if reset_button:
                    st.session_state.heatmap_accumulator = np.zeros_like(st.session_state.heatmap_accumulator)
                    st.session_state.frame_count = 0
        # No controls needed here for prerendered mode as speed controls are in sidebar

    with row2_col3:
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
                    <td>Diameter</td>
                    <td class="highlight-value">10.2 mm</td>
                    <td class="positive-value">7.5 mm</td>
                </tr>
                <tr>
                    <td>Flow</td>
                    <td class="highlight-value">8.4 cm/s</td>
                    <td class="positive-value">19.3 cm/s</td>
                </tr>
                <tr>
                    <td>Augmentation</td>
                    <td class="highlight-value">5%</                    <td class="positive-value">45%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Bottom Row - Actions and Additional Data
    with row3_col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-header">Risk Factors</div>
            <ul>
                <li><span class="highlight-value">Wells Score: 6 (High)</span></li>
                <li>D-dimer: 1450 ng/mL</li>
                <li>Immobilization >3 days</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with row3_col2:
        # Recommended actions
        st.markdown("""
        <div class="info-card">
            <div class="card-header">Recommended Actions</div>
            <ol>
                <li><span class="highlight-value">Urgent vascular consult</span></li>
                <li>Confirmatory Doppler ultrasound</li>
                <li>Consider anticoagulants (medium dose)</li>
                <li>Monitor for PE symptoms: Dyspnea, chest pain, hemoptysis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with row3_col3:
        st.markdown("""
        <div class="info-card">
            <div class="card-header">Analysis History</div>
            <div style="font-size: 0.9em;">
                <p><strong>Current:</strong> <span class="highlight-value">DVT Detected</span></p>
                <p><strong>02/05/2025:</strong> <span class="positive-value">Normal</span></p>
                <p><strong>01/12/2025:</strong> <span class="positive-value">Normal</span></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Tabs for detailed analysis ---
    st.markdown("<h3>Detailed Analysis</h3>", unsafe_allow_html=True)
    tabs = st.tabs(["Flow Analysis", "Compression Test", "Vessel Comparison"])

    with tabs[0]:
        flow_col1, flow_col2 = st.columns([2, 1])
        
        with flow_col1:
            # Doppler Spectral Waveform
            st.caption("Doppler Spectral Waveform Analysis")
            
            # Create subplot with shared x-axis
            fig_doppler = make_subplots(rows=2, cols=1, shared_xaxes=True)
            
            # Time points
            x_time = np.linspace(0, 4, 200)
            
            # Normal vein data
            normal_data = 20 + 12 * np.sin(2*np.pi*x_time) + 5 * np.sin(6*np.pi*x_time)
            
            # Abnormal DVT data
            dvt_data = 10 + 4 * np.sin(2*np.pi*x_time) + np.random.normal(0, 1, 200)
            
            fig_doppler.add_trace(
                go.Scatter(x=x_time, y=normal_data, mode='lines', name='Normal Vein',
                          line=dict(color='rgba(0,100,255,0.8)')),
                row=1, col=1
            )
            
            fig_doppler.add_trace(
                go.Scatter(x=x_time, y=dvt_data, mode='lines', name='DVT Affected Vein',
                          line=dict(color='rgba(255,0,0,0.8)')),
                row=2, col=1
            )
            
            fig_doppler.update_layout(height=400, title_text="Spectral Doppler Comparison")
            fig_doppler.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig_doppler.update_yaxes(title_text="Flow Velocity (cm/s)", row=1, col=1)
            fig_doppler.update_yaxes(title_text="Flow Velocity (cm/s)", row=2, col=1)
            
            flow_chart = st.plotly_chart(fig_doppler, use_container_width=True)
        
        with flow_col2:
            # Key metrics table
            st.subheader("Flow Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Peak Systolic Velocity', 'End Diastolic Velocity', 'Resistance Index', 'Augmentation'],
                'Value': ['8.4 cm/s', '4.2 cm/s', '0.5', 'Absent'],
                'Normal Range': ['15-25 cm/s', '5-10 cm/s', '0.8-1.0', 'Present'],
                'Status': ['Abnormal ⚠️', 'Abnormal ⚠️', 'Abnormal ⚠️', 'Abnormal ⚠️']
            })
            metrics_table = st.table(metrics_df)
            
            st.info("Flow analysis shows significantly reduced velocity and loss of normal phasicity, consistent with venous obstruction.")

    with tabs[1]:
        # Compression Ultrasound Test
        st.caption("Compression Ultrasound Assessment")
        
        # Create visualization of vessel compression test
        fig_compression = go.Figure()
        
        x_pos = np.linspace(0, 10, 100)
        no_compression = np.ones(100) * 10
        
        partial_compression = np.ones(100) * 4
        partial_compression[30:70] = 8 - 4*np.exp(-((x_pos[30:70]-5)**2)/2)
        
        full_compression = np.ones(100) * 1
        
        fig_compression.add_trace(go.Scatter(x=x_pos, y=no_compression, mode='lines', name='No Compression',
                               line=dict(color='blue')))
        fig_compression.add_trace(go.Scatter(x=x_pos, y=partial_compression, mode='lines', name='With Compression (DVT)',
                               line=dict(color='red')))
        fig_compression.add_trace(go.Scatter(x=x_pos, y=full_compression, mode='lines', name='Normal Response',
                               line=dict(color='green', dash='dash')))
        
        fig_compression.update_layout(title='Vessel Compressibility Test',
                         xaxis_title='Position Along Vessel (cm)',
                         yaxis_title='Vessel Diameter (mm)')
        
        compression_chart = st.plotly_chart(fig_compression, use_container_width=True)
        
        compression_info = st.info("**Interpretation**: Incomplete compression of the vein is highly suggestive of DVT. " 
                "Normal veins should completely collapse under probe pressure.")

    with tabs[2]:
        # Vessel comparison
        st.caption("Bilateral Vessel Comparison")
        
        comparison_data = {
            'Measurement': ['Vessel Diameter (mm)', 'Peak Flow Velocity (cm/s)', 'Augmentation Response (%)', 
                           'Spontaneous Flow', 'Echogenicity'],
            'Affected Side': ['10.2', '8.4', '5', 'Absent', 'Hyperechoic'],
            'Contralateral': ['7.5', '19.3', '45', 'Present', 'Normal']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_table = st.table(comparison_df)
        
        # Bar chart for comparison
        fig_comparison = go.Figure()
        
        categories = ['Vessel Diameter (mm)', 'Peak Flow (cm/s)', 'Augmentation (%)']
        affected = [10.2, 8.4, 5]
        normal = [7.5, 19.3, 45]
        
        fig_comparison.add_trace(go.Bar(x=categories, y=affected, name='Affected Side', marker_color='red'))
        fig_comparison.add_trace(go.Bar(x=categories, y=normal, name='Contralateral (Normal) Side', marker_color='blue'))
        
        fig_comparison.update_layout(title='Bilateral Comparison of Key Metrics',
                         xaxis_title='Measurement',
                         yaxis_title='Value',
                         barmode='group')
        
        comparison_chart = st.plotly_chart(fig_comparison, use_container_width=True)

    # --- Now run the video processing loop ---
    def process_frame():
        # Add error handling to the frame processing function
        try:
            ret, frame = cap.read()
            if not ret:
                # Try to reset to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    # If still can't read frame, return None to signal error
                    return None
            
            # Increment frame count
            st.session_state.frame_count += 1
            
            # If in prerendered mode, just show the frame directly
            if demo_mode == "Prerendered":
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # For Live mode, create a copy of the frame for heatmap
            display_frame = frame.copy()
            
            # Run inference on the frame using the model
            if model is not None:
                results = model(frame, stream=True)
                
                # Process detection results
                detections_this_frame = np.zeros_like(st.session_state.heatmap_accumulator)
                
                # Draw bounding boxes and update heatmap
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Draw rectangle on display frame
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence text
                        text = f"{confidence:.2f}"
                        cv2.putText(display_frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # If heatmap is active, update the detection for this frame
                        if st.session_state.heatmap_activated:
                            # Create heatmap intensity based on confidence
                            intensity = confidence * 255
                            
                            # Add a Gaussian blob centered on the detection
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Size the Gaussian based on detection size
                            sigma_x = max(2, (x2 - x1) // 4)
                            sigma_y = max(2, (y2 - y1) // 4)
                            
                            # Create a mask for the current detection
                            y_coords, x_coords = np.ogrid[:detections_this_frame.shape[0], :detections_this_frame.shape[1]]
                            
                            # Calculate distance from center (vectorized)
                            dist_from_center = ((x_coords - center_x) ** 2) / (2 * sigma_x ** 2) + \
                                               ((y_coords - center_y) ** 2) / (2 * sigma_y ** 2)
                            
                            # Create Gaussian mask
                            mask = intensity * np.exp(-dist_from_center)
                            
                            # Update the detection frame
                            detections_this_frame = np.maximum(detections_this_frame, mask)
                
                # Update the heatmap accumulator if activated
                if st.session_state.heatmap_activated:
                    # Apply decay factor to existing heatmap
                    decay_factor = 0.95
                    st.session_state.heatmap_accumulator *= decay_factor
                    
                    # Add new detections to accumulator
                    st.session_state.heatmap_accumulator = np.maximum(
                        st.session_state.heatmap_accumulator, 
                        detections_this_frame
                    )
                    
                    # Apply the heatmap to the display frame
                    if np.max(st.session_state.heatmap_accumulator) > 0:
                        normalized_heatmap = cv2.normalize(
                            st.session_state.heatmap_accumulator, 
                            None, 
                            0, 
                            255, 
                            cv2.NORM_MINMAX
                        ).astype(np.uint8)
                        
                        # Apply colormap
                        colormap = getattr(cv2, selected_colormap)
                        colored_heatmap = cv2.applyColorMap(normalized_heatmap, colormap)
                        
                        # Blend with original frame
                        alpha = heatmap_intensity
                        beta = 1 - alpha
                        display_frame = cv2.addWeighted(display_frame, beta, colored_heatmap, alpha, 0)
            
            # Convert color format for Streamlit display
            return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.error(f"Error processing frame: {e}")
            return None

    # Run the video processing loop with error handling
    if video_placeholder:
        try:
            while True:
                # Get and process a new frame
                processed_frame = process_frame()
                if processed_frame is not None:
                    # Update the video display
                    video_placeholder.image(processed_frame, channels="RGB")
                    
                    # Control the inference/playback speed based on mode
                    if demo_mode == "Live":
                        time.sleep(1.0/inference_speed)
                    else:
                        # Use the selected playback speed for prerendered videos
                        time.sleep(1.0/(30.0 * st.session_state.playback_speed))  # Assuming 30 fps base
                else:
                    st.warning("End of video or frame processing error. Please select another video.")
                    break
        except Exception as e:
            st.error(f"Video playback error: {e}")
            st.code(traceback.format_exc())
