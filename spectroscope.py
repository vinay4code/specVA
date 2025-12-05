import streamlit as st
import numpy as np
from astropy.io import fits
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from io import BytesIO
import time

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
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
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

# --- SESSION STATE INITIALIZATION ---
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# --- CONSTANTS & REFERENCE DATA ---
COMMON_LINES = {
    "Hydrogen Balmer": {"H-alpha": 6562.8, "H-beta": 4861.3, "H-gamma": 4340.5, "H-delta": 4101.7},
    "Helium (He I)": {"He I (Yellow)": 5875.6, "He I (Blue)": 4471.5},
    "Sodium (Na)": {"Na D1": 5895.9, "Na D2": 5889.9},
    "Oxygen (O III)": {"O III (Nebular)": 5006.8}
}

# --- UTILITY FUNCTIONS ---
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# --- SIDEBAR UI ---
with st.sidebar:
    try:
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
    
    with st.expander("**Calibration Settings**", expanded=True):
        st.info("Map pixels to Angstroms")
        cal_mode = st.radio("Mode", ["Manual Calibration", "Auto (From Header)"])
        
        start_wavelength = 0.0
        dispersion = 1.0
        
        if cal_mode == "Manual Calibration":
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
# Using columns to create a balanced header
col_logo, col_text = st.columns([1, 5])

with col_text:
    st.title("**N-SIGHT**")
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1F2937, transparent); padding: 10px; border-radius: 5px; border-left: 5px solid #FF4B4B;'>
    <b>Telescope Team Project</b> | National Institute of Technology, Trichy
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- DATA INPUT SECTION ---
# Wrapped in a container for styling
with st.container():
    st.subheader("Input Data")
    uploaded_file = st.file_uploader("Drop your FITS file to begin...", type=["fit", "fits"])

# --- MAIN LOGIC ---
data = None
header = None
is_demo = False

if uploaded_file is not None:
    st.session_state.demo_mode = False 
    try:
        with st.spinner("Decoding FITS file..."):
            with fits.open(uploaded_file) as hdul:
                data = hdul[0].data
                header = hdul[0].header
            # Artificial delay for UX "processing" feel
            time.sleep(0.5)
    except Exception as e:
        st.error(f"Error reading file: {e}")

elif st.session_state.demo_mode:
    is_demo = True
    # Generate demo data
    x_demo = np.linspace(4000, 7000, 2000) 
    continuum = 100 + (x_demo - 4000) * 0.05
    h_alpha = 500 * np.exp(-0.5 * ((x_demo - 6563) / 10)**2)
    h_beta = 300 * np.exp(-0.5 * ((x_demo - 4861) / 10)**2)
    noise = np.random.normal(0, 5, 2000)
    data = continuum + h_alpha + h_beta + noise
    
    header = {'TELESCOP': 'Simulated Scope', 'CRVAL1': 4000.0, 'CDELT1': 1.5}
    
    st.info("**Demo Mode Active**")

# --- VISUALIZATION & ANALYSIS ---
if data is not None and header is not None:
    # 1. Image Processing (if 2D)
    if data.ndim == 2:
        st.subheader("Raw CCD Sensor")
        col_img, col_info = st.columns([3, 1])
        with col_img:
            st.image(normalize_data(data), caption="Spectral Stripe", use_container_width=True, clamp=True)
        with col_info:
            st.success("2D Spectrum Detected")
            st.markdown("Integrating vertical pixels to extract intensity profile.")
        flux = np.sum(data, axis=0)
    elif data.ndim == 1:
        flux = data
    else:
        st.error("Invalid dimensions.")
        st.stop()

    # 2. Data Transforms
    if normalize:
        flux = normalize_data(flux)
    
    if smoothing_window > 1:
        if smoothing_window % 2 == 0: smoothing_window += 1
        flux = savgol_filter(flux, smoothing_window, 3)

    # 3. Calibration Logic
    pixels = np.arange(len(flux))
    
    if cal_mode == "Auto (From Header)":
        try:
            start_wavelength = header['CRVAL1']
            dispersion = header['CDELT1']
            if 'cal_success' not in st.session_state:
                st.toast(f"Calibration Found: {start_wavelength}Å + {dispersion}Å/px", icon="✅")
                st.session_state.cal_success = True
        except KeyError:
            st.warning("Header missing calibration keywords. Using manual settings.")
            cal_mode = "Manual Calibration"

    if cal_mode == "Manual Calibration" or cal_mode == "Pixel Space":
        x_axis = start_wavelength + (pixels * dispersion)
        x_label = "Wavelength (Å)"
    
    # 4. Plotting
    peak_indices = []
    if show_peaks:
        peak_indices, _ = find_peaks(flux, prominence=peak_prominence)

    st.subheader("Spectral Analysis")
    
    # Create the Plotly Figure
    fig = go.Figure()

    # Main Line with a "Neon" look
    fig.add_trace(go.Scatter(
        x=x_axis, y=flux, mode='lines', name='Spectrum',
        line=dict(color='#00F0FF', width=2, shape='hv'), # Cyan neon color
        fill='tozeroy', fillcolor='rgba(0, 240, 255, 0.1)' # Subtle glow fill
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

    # High-Tech Chart Layout
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
    
    # 5. Footer / Header Details
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.expander("View FITS Header Metadata"):
            header_dict = dict(header) if hasattr(header, 'items') else {k: str(v) for k, v in header.items()}
            st.table(header_dict)
    
    with col2:
        if is_demo:
            if st.button("Exit Demo", use_container_width=True):
                st.session_state.demo_mode = False
                st.rerun()

else:
    # --- LANDING PAGE (Empty State) ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_center, _ = st.columns([1, 0.1])
    with col_center:
        st.info("Please upload a FITS file above to unlock analysis tools.")
    
    st.markdown("---")
    st.subheader("No Data? Try the Simulation")
    
    # Demo Generation
    x_gen = np.linspace(4000, 7000, 2000) 
    continuum_gen = 100 + (x_gen - 4000) * 0.05
    h_alpha_gen = 500 * np.exp(-0.5 * ((x_gen - 6563) / 10)**2)
    h_beta_gen = 300 * np.exp(-0.5 * ((x_gen - 4861) / 10)**2)
    noise_gen = np.random.normal(0, 5, 2000)
    y_gen = continuum_gen + h_alpha_gen + h_beta_gen + noise_gen
    
    hdu_gen = fits.PrimaryHDU(y_gen)
    hdu_gen.header['TELESCOP'] = 'Simulated Scope'
    bio_gen = BytesIO()
    hdu_gen.writeto(bio_gen)
    bio_gen.seek(0)
    
    # Action Buttons
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        if st.button("Load Demo", use_container_width=True):
            st.session_state.demo_mode = True
            st.rerun()
            
    with c2:
        st.download_button(
            label="Save FITS",
            data=bio_gen,
            file_name="demo_spectrum.fits",
            mime="application/octet-stream",
            use_container_width=True
        )
