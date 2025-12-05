import streamlit as st
import numpy as np
import cv2
from astropy.io import fits
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from io import BytesIO
import time
import struct
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Nakshatra N-SIGHT",
    layout="wide",
    page_icon="🔭",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS & ANIMATIONS ---
def apply_custom_style():
    st.markdown("""
        <style>
        /* MAIN BACKGROUND & FONT */
        .stApp {
            background-color: #0E1117;
        }
        
        /* KEYFRAME ANIMATIONS */
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        /* APPLY ANIMATION TO MAIN BLOCKS */
        .main .block-container {
            animation: fadeIn 0.8s ease-out;
        }

        /* CUSTOM BUTTON STYLING */
        .stButton>button {
            background: linear-gradient(90deg, #FF4B4B 0%, #FF914D 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-weight: bold;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 20px rgba(255, 75, 75, 0.3);
        }
        .stButton>button:active {
            transform: scale(0.95);
        }

        /* UPLOAD WIDGET STYLING */
        [data-testid='stFileUploader'] {
            border: 2px dashed #4B5563;
            border-radius: 15px;
            padding: 20px;
            background-color: #1F2937;
            transition: border-color 0.3s;
        }
        [data-testid='stFileUploader']:hover {
            border-color: #FF4B4B;
        }

        /* EXPANDER STYLING */
        .streamlit-expanderHeader {
            background-color: #1F2937;
            border-radius: 10px;
            color: #E5E7EB;
        }
        
        /* DATAFRAME/TABLE STYLING */
        [data-testid="stTable"] {
            background-color: #1F2937;
            border-radius: 10px;
            padding: 10px;
        }
        
        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #374151;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# --- HELPER CLASS: SER READER (Added Feature 1) ---
class SERReader:
    """
    Reads SER video files (common in Planetary Astrophotography).
    """
    def __init__(self, file_buffer):
        self.file = file_buffer
        self.header = {}
        self._parse_header()

    def _parse_header(self):
        self.file.seek(0)
        # SER Header is 178 bytes
        header_data = self.file.read(178)
        
        self.header['FileID'] = header_data[0:14].decode('utf-8').strip()
        self.header['LuID'] = struct.unpack('<I', header_data[14:18])[0]
        self.header['ColorID'] = struct.unpack('<I', header_data[18:22])[0] # 0=Mono, 8-11=Bayer, 100=RGB
        self.header['LittleEndian'] = struct.unpack('<I', header_data[22:26])[0]
        self.header['Width'] = struct.unpack('<I', header_data[26:30])[0]
        self.header['Height'] = struct.unpack('<I', header_data[30:34])[0]
        self.header['PixelDepth'] = struct.unpack('<I', header_data[34:38])[0]
        self.header['FrameCount'] = struct.unpack('<I', header_data[38:42])[0]
        
        self.bytes_per_pixel = 1 if self.header['PixelDepth'] <= 8 else 2
        self.frame_size = self.header['Width'] * self.header['Height'] * self.bytes_per_pixel

    def get_frame(self, frame_index):
        if frame_index >= self.header['FrameCount']: return None
        
        offset = 178 + (frame_index * self.frame_size)
        self.file.seek(offset)
        data = self.file.read(self.frame_size)
        
        dtype = np.uint8 if self.bytes_per_pixel == 1 else np.uint16
        image_data = np.frombuffer(data, dtype=dtype)
        # Reshape and return
        return image_data.reshape((self.header['Height'], self.header['Width']))

# --- SESSION STATE INITIALIZATION ---
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False
if 'captured_data' not in st.session_state: 
    st.session_state.captured_data = None
if 'run_cam' not in st.session_state:
    st.session_state.run_cam = False

# --- CONSTANTS & REFERENCE DATA ---
COMMON_LINES = {
    "Hydrogen Balmer": {"H-alpha": 6562.8, "H-beta": 4861.3, "H-gamma": 4340.5, "H-delta": 4101.7},
    "Helium (He I)": {"He I (Yellow)": 5875.6, "He I (Blue)": 4471.5},
    "Sodium (Na)": {"Na D1": 5895.9, "Na D2": 5889.9},
    "Oxygen (O III)": {"O III (Nebular)": 5006.8}
}

# --- UTILITY FUNCTIONS ---
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)

# --- SIDEBAR UI ---
with st.sidebar:
    try:
        # Try to load the logo image
        st.image("Nakshatra_transparent_1.png", use_container_width=True)
    except Exception:
        # Fallback text logo if image fails
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='margin:0; color: #FF4B4B;'>🔭</h1>
            <h3 style='margin:0;'>N-SIGHT</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### **Nakshatra Club NITT**")
    st.caption("Telescope Team • Spectroscopy Division")
    st.markdown("---")
    
    # --- NEW: INPUT SOURCE SELECTOR ---
    input_method = st.radio("Select Input Source:", 
                           ["Upload File (FITS/SER)", "Live Camera (QHY/USB)", "Demo Simulation"])
    st.markdown("---")
    
    with st.expander("**Calibration Settings**", expanded=True):
        st.info("Map pixels to Angstroms")
        cal_mode = st.radio("Mode", ["Manual Calibration", "Auto (From Header)"])
        
        start_wavelength = 0.0
        dispersion = 1.0
        
        col1, col2 = st.columns(2)
        with col1:
            start_wavelength = st.number_input("Start (Å)", value=4000.0, step=10.0)
        with col2:
            dispersion = st.number_input("Disp (Å/px)", value=1.5, step=0.01)
    
    with st.expander("🔍 **Analysis Tools**", expanded=True):
        smoothing_window = st.slider("Smoothing (Noise)", min_value=1, max_value=51, value=1, step=2)
        
        show_peaks = st.checkbox("Auto-Detect Peaks", value=False)
        peak_prominence = 10.0
        if show_peaks:
            peak_prominence = st.slider("Peak Sensitivity", 0.1, 50.0, 10.0)
        
        st.markdown("**Reference Overlays**")
        show_ref_lines = st.multiselect("Select Elements", options=list(COMMON_LINES.keys()), default=[])

    with st.expander("🎨 **Display Options**"):
        show_grid = st.checkbox("Show Grid", value=True)
        normalize = st.checkbox("Normalize (0-1)", value=False)
        invert_yaxis = st.checkbox("Invert Y-Axis", value=False)

# --- HERO SECTION ---
col_logo, col_text = st.columns([1, 5])

with col_text:
    st.title("**N-SIGHT**")
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1F2937, transparent); padding: 10px; border-radius: 5px; border-left: 5px solid #FF4B4B;'>
    <b>Telescope Team Project</b> | National Institute of Technology, Trichy
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- DATA INPUT LOGIC ---
data = None
header = {}
is_demo = False

# 1. FILE UPLOAD (FITS + SER)
if input_method == "Upload File (FITS/SER)":
    uploaded_file = st.file_uploader("Drop your FITS or SER file...", type=["fit", "fits", "ser"])
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # Handle FITS
        if file_ext in ['fit', 'fits']:
            try:
                with st.spinner("Decoding FITS file..."):
                    with fits.open(uploaded_file) as hdul:
                        data = hdul[0].data
                        header = dict(hdul[0].header)
            except Exception as e:
                st.error(f"Error reading FITS: {e}")
                
        # Handle SER (New Feature 1)
        elif file_ext == 'ser':
            try:
                with st.spinner("Decoding SER video..."):
                    reader = SERReader(uploaded_file)
                    st.success(f"SER Loaded: {reader.header['FrameCount']} frames | {reader.header['Width']}x{reader.header['Height']}")
                    
                    # SER Controls
                    ser_mode = st.radio("SER Processing Mode", ["Single Frame", "Stack All Frames (Average)"], horizontal=True)
                    
                    if ser_mode == "Single Frame":
                        f_idx = st.slider("Frame Index", 0, reader.header['FrameCount']-1, 0)
                        data = reader.get_frame(f_idx)
                    else:
                        if st.button("Stack Frames"):
                            frames = []
                            # Stack first 100 frames max to avoid memory overload in browser/demo
                            count = min(100, reader.header['FrameCount'])
                            progress_bar = st.progress(0)
                            for i in range(count):
                                frames.append(reader.get_frame(i))
                                progress_bar.progress((i + 1) / count)
                            data = np.mean(frames, axis=0)
                            st.info(f"Stacked {count} frames.")
                            
                    header = {"TELESCOP": "SER Source", "INSTRUME": "QHY/CMOS"}
            except Exception as e:
                st.error(f"Error reading SER: {e}")

# 2. LIVE CAMERA (New Feature 2)
elif input_method == "Live Camera (QHY/USB)":
    st.subheader("Live Spectral Feed")
    cam_type = st.radio("Camera Driver", ["Browser Webcam (Basic)", "Local USB (OpenCV/QHY)"], horizontal=True)
    
    if cam_type == "Browser Webcam (Basic)":
        cam_input = st.camera_input("Capture Spectrum")
        if cam_input:
            img = Image.open(cam_input).convert('L')
            data = np.array(img)
            header = {"TELESCOP": "Browser Cam"}
            
    else:
        # OpenCV Local Camera Logic
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            cam_idx = st.number_input("Camera Index", 0, 10, 0)
            if st.button("Initialize Camera"):
                st.session_state.run_cam = True
        
        with col_c2:
            if st.session_state.run_cam:
                frame_placeholder = st.empty()
                stop_btn = st.button("Stop & Use Last Frame")
                capture_btn = st.button("Capture & Analyze")
                
                cap = cv2.VideoCapture(cam_idx)
                # Force High Res for QHY
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                
                if not cap.isOpened():
                    st.error(f"Cannot open camera {cam_idx}. Ensure WDM drivers are installed.")
                else:
                    while cap.isOpened() and st.session_state.run_cam:
                        ret, frame = cap.read()
                        if not ret: break
                        
                        # Display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB")
                        
                        if capture_btn:
                            st.session_state.captured_data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            st.session_state.run_cam = False
                            cap.release()
                            st.rerun()
                        
                        if stop_btn:
                            st.session_state.run_cam = False
                            cap.release()
                            break
                    cap.release()

    if st.session_state.captured_data is not None:
        data = st.session_state.captured_data
        header = {"TELESCOP": "Live QHY/USB"}
        st.success("Frame Captured!")
        if st.button("Clear Capture"):
            st.session_state.captured_data = None
            st.rerun()

# 3. DEMO SIMULATION
elif input_method == "Demo Simulation":
    is_demo = True
    x_demo = np.linspace(4000, 7000, 2000)
    continuum = 100 + (x_demo - 4000) * 0.05
    h_alpha = 500 * np.exp(-0.5 * ((x_demo - 6563) / 10)**2)
    h_beta = 300 * np.exp(-0.5 * ((x_demo - 4861) / 10)**2)
    noise = np.random.normal(0, 5, 2000)
    data = continuum + h_alpha + h_beta + noise
    header = {'TELESCOP': 'Simulated Scope', 'CRVAL1': 4000.0, 'CDELT1': 1.5}
    st.info("**Demo Mode Active**")


# --- PROCESSING & VISUALIZATION ---
if data is not None:
    
    # --- 2D IMAGE PROCESSING ---
    if data.ndim == 2:
        st.subheader("Raw Sensor Data")
        
        # Interactive Crop/Scan
        col_img, col_info = st.columns([3, 1])
        h, w = data.shape
        
        with col_info:
            st.markdown("### Region of Interest")
            scan_y = st.slider("Scan Line Y", 0, h-1, h//2)
            scan_height = st.slider("Band Height", 1, 100, 10)
        
        with col_img:
            # Display Image with Overlay
            display_img = normalize_data(data)
            display_rgb = np.stack([display_img]*3, axis=-1)
            
            y_start = max(0, scan_y - scan_height//2)
            y_end = min(h, scan_y + scan_height//2)
            
            # Plotly Image for faster rendering
            fig_img = go.Figure(go.Image(z=(display_rgb * 255).astype(np.uint8)))
            fig_img.add_shape(type="rect", x0=0, y0=y_start, x1=w, y1=y_end, line=dict(color="red", width=2))
            fig_img.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_img, use_container_width=True)
            
        flux = np.mean(data[y_start:y_end, :], axis=0)
        
    elif data.ndim == 1:
        flux = data
    else:
        st.error("Invalid data dimensions")
        st.stop()

    # --- DATA TRANSFORMS ---
    if normalize:
        flux = normalize_data(flux)
    
    if smoothing_window > 1:
        if smoothing_window % 2 == 0: smoothing_window += 1
        flux = savgol_filter(flux, smoothing_window, 3)

    # --- CALIBRATION LOGIC ---
    pixels = np.arange(len(flux))
    
    # Updated Logic: Update the variables if Auto is selected...
    if cal_mode == "Auto (From Header)":
        start_wavelength = header.get('CRVAL1', start_wavelength)
        dispersion = header.get('CDELT1', dispersion)

    # ... But ALWAYS calculate x_axis using the current values (whether updated or default/manual)
    x_axis = start_wavelength + (pixels * dispersion)
    x_label = "Wavelength (Å)"
    
    # --- PLOTTING ---
    peak_indices = []
    if show_peaks:
        peak_indices, _ = find_peaks(flux, prominence=peak_prominence)

    st.subheader("Spectral Analysis")
    
    fig = go.Figure()

    # Main Spectrum Trace (Neon Style)
    fig.add_trace(go.Scatter(
        x=x_axis, y=flux, mode='lines', name='Spectrum',
        line=dict(color='#00F0FF', width=2, shape='hv'), # Cyan neon color
        fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.1)'
    ))

    # Peaks
    if show_peaks and len(peak_indices) > 0:
        fig.add_trace(go.Scatter(
            x=x_axis[peak_indices], y=flux[peak_indices], mode='markers',
            name='Peaks', marker=dict(color='#FF4B4B', size=10, symbol='diamond-open', line=dict(width=2))
        ))
        for i in peak_indices:
            fig.add_annotation(
                x=x_axis[i], y=flux[i], text=f"{x_axis[i]:.0f}Å",
                showarrow=True, arrowhead=1, yshift=15, font=dict(color='#FF4B4B')
            )

    # Reference Lines
    for element_group in show_ref_lines:
        lines = COMMON_LINES[element_group]
        for name, wl in lines.items():
            if x_axis.min() < wl < x_axis.max():
                fig.add_vline(x=wl, line_width=1, line_dash="dot", line_color="#FFD700")
                fig.add_annotation(
                    x=wl, y=0.98 if not invert_yaxis else 0.02, yref="paper",
                    text=name, showarrow=False, font=dict(color="#FFD700", size=10), textangle=-90
                )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title=x_label, showgrid=show_grid, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="Intensity", showgrid=show_grid, gridcolor='rgba(255,255,255,0.1)'),
        height=600,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.1)
    )

    if invert_yaxis:
        fig['layout']['yaxis']['autorange'] = "reversed"

    st.plotly_chart(fig, use_container_width=True)
    
    # --- FOOTER / METADATA ---
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.expander("View Header Metadata"):
            header_dict = dict(header) if hasattr(header, 'items') else {k: str(v) for k, v in header.items()}
            st.table(header_dict)
    
    with col2:
        st.markdown("### Export")
        csv_text = "Wavelength,Intensity\n" + "\n".join([f"{x:.2f},{y:.2f}" for x, y in zip(x_axis, flux)])
        st.download_button(
            label="Download CSV",
            data=csv_text,
            file_name="spectrum_data.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # --- EMPTY STATE ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👈 Select 'Demo Simulation' or Upload a File in the sidebar to begin.")
