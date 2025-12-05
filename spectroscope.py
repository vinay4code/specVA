import streamlit as st
import numpy as np
from astropy.io import fits
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from io import BytesIO

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Nakshatra N-SIGHT",
    layout="wide", 
    page_icon="🔭",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE INITIALIZATION ---
# This keeps the demo data loaded even after you click buttons
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# --- CONSTANTS & REFERENCE DATA ---
COMMON_LINES = {
    "Hydrogen Balmer": {
        "H-alpha": 6562.8,
        "H-beta": 4861.3,
        "H-gamma": 4340.5,
        "H-delta": 4101.7
    },
    "Helium (He I)": {
        "He I (Yellow)": 5875.6,
        "He I (Blue)": 4471.5
    },
    "Sodium (Na)": {
        "Na D1": 5895.9,
        "Na D2": 5889.9
    },
    "Oxygen (O III)": {
        "O III (Nebular)": 5006.8
    }
}

# --- UTILITY FUNCTIONS ---
def normalize_data(data):
    """Normalize data between 0 and 1."""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# --- SIDEBAR UI ---
with st.sidebar:
    try:
        st.image("Nakshatra_transparent_1.png", use_container_width=True)
    except Exception:
        st.warning("⚠️ Logo 'Nakshatra_transparent_1.png' not found.")

    st.header("Nakshatra Club NITT") 
    st.caption("Telescope Team • Spectroscopy Division")
    st.divider()
    
    st.header("Calibration")
    st.info("Map pixels to Angstroms")
    
    cal_mode = st.radio("Mode", ["Manual Calibration", "Auto (From Header)"])
    
    start_wavelength = 0.0
    dispersion = 1.0
    
    if cal_mode == "Manual Calibration":
        col1, col2 = st.columns(2)
        with col1:
            start_wavelength = st.number_input("Start Wavelength (Å)", value=4000.0, step=10.0)
        with col2:
            dispersion = st.number_input("Dispersion (Å/px)", value=1.5, step=0.01)
    
    st.divider()
    
    st.header("Analysis Tools")
    
    smoothing_window = st.slider("Noise Reduction (Smoothing)", min_value=1, max_value=51, value=1, step=2)
    
    show_peaks = st.checkbox("Auto-Detect Peaks", value=False)
    if show_peaks:
        peak_prominence = st.slider("Peak Sensitivity", 0.1, 50.0, 10.0)
    
    st.subheader("Overlays")
    show_ref_lines = st.multiselect("Show Reference Lines", options=list(COMMON_LINES.keys()), default=[])

    st.divider()
    
    st.header("Display Options")
    show_grid = st.checkbox("Show Grid", value=True)
    normalize = st.checkbox("Normalize Intensity (0-1)", value=False)
    invert_yaxis = st.checkbox("Invert Y-Axis (Magnitudes)", value=False)

# --- MAIN PAGE HEADER ---
st.title("**N-SIGHT**")
st.markdown("Welcome to N-SIGHT. This tool allows the Telescope Team to analyze spectral data.")
st.markdown("---")

# --- DATA INPUT SECTION ---
st.subheader("Data Input")
uploaded_file = st.file_uploader("Upload your FITS file here to begin analysis", type=["fit", "fits"])

# --- MAIN LOGIC ---

# 1. Determine if we are using Uploaded File or Demo Data
data = None
header = None
is_demo = False

if uploaded_file is not None:
    # Prioritize Upload
    st.session_state.demo_mode = False 
    try:
        with fits.open(uploaded_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header
    except Exception as e:
        st.error(f"Error reading file: {e}")

elif st.session_state.demo_mode:
    # Use Demo Data if active
    is_demo = True
    x_demo = np.linspace(4000, 7000, 2000) 
    continuum = 100 + (x_demo - 4000) * 0.05
    h_alpha = 500 * np.exp(-0.5 * ((x_demo - 6563) / 10)**2)
    h_beta = 300 * np.exp(-0.5 * ((x_demo - 4861) / 10)**2)
    noise = np.random.normal(0, 5, 2000)
    data = continuum + h_alpha + h_beta + noise
    
    # Fake header for demo
    header = {
        'TELESCOP': 'Simulated Scope',
        'CRVAL1': 4000.0,
        'CDELT1': 1.5
    }
    st.info("ℹ️ Viewing **Demo Data**. Click 'Exit Demo Mode' below to clear.")

# 2. Process Data if it exists
if data is not None and header is not None:
    # --- DATA PROCESSING ---
    if data.ndim == 2:
        st.subheader("Raw Sensor Data")
        col_img, col_desc = st.columns([3, 1])
        with col_img:
            st.image(normalize_data(data), caption="Raw CCD Image", use_container_width=True, clamp=True)
        with col_desc:
            st.info("Converting 2D image to 1D profile by integrating vertically.")
        flux = np.sum(data, axis=0)
    elif data.ndim == 1:
        flux = data
    else:
        st.error("Data dimensions not supported.")
        st.stop()

    if normalize:
        flux = normalize_data(flux)
        
    if smoothing_window > 1:
        if smoothing_window % 2 == 0: smoothing_window += 1
        flux = savgol_filter(flux, smoothing_window, 3)

    # --- WAVELENGTH CALIBRATION ---
    pixels = np.arange(len(flux))
    
    if cal_mode == "Auto (From Header)":
        try:
            start_wavelength = header['CRVAL1']
            dispersion = header['CDELT1']
            st.sidebar.success(f"Found Header Calibration: Start={start_wavelength}, Disp={dispersion}")
        except KeyError:
            st.sidebar.error("Header info (CRVAL1/CDELT1) not found. Switching to Manual.")
            cal_mode = "Manual Calibration"

    if cal_mode == "Manual Calibration" or cal_mode == "Pixel Space":
        x_axis = start_wavelength + (pixels * dispersion)
        x_label = "Wavelength (Angstroms)"
    
    # --- PEAK DETECTION ---
    peak_indices = []
    if show_peaks:
        peak_indices, _ = find_peaks(flux, prominence=peak_prominence)

    # --- PLOTTING ---
    st.subheader("Spectral Analysis")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_axis, y=flux, mode='lines', name='Spectrum Data',
        line=dict(color='#00CCFF', width=2)
    ))

    if show_peaks and len(peak_indices) > 0:
        fig.add_trace(go.Scatter(
            x=x_axis[peak_indices], y=flux[peak_indices], mode='markers',
            name='Detected Peaks', marker=dict(color='red', size=8, symbol='x')
        ))
        for i in peak_indices:
            fig.add_annotation(
                x=x_axis[i], y=flux[i], text=f"{x_axis[i]:.1f}Å",
                showarrow=True, arrowhead=1, yshift=10
            )

    for element_group in show_ref_lines:
        lines = COMMON_LINES[element_group]
        for name, wl in lines.items():
            if x_axis.min() < wl < x_axis.max():
                fig.add_vline(x=wl, line_width=1, line_dash="dash", line_color="yellow")
                fig.add_annotation(
                    x=wl, y=0.95 if not invert_yaxis else 0.05, yref="paper",
                    text=name, showarrow=False, font=dict(color="yellow"), textangle=-90
                )

    fig.update_layout(
        template="plotly_dark", xaxis_title=x_label, yaxis_title="Intensity",
        height=600, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if invert_yaxis:
        fig['layout']['yaxis']['autorange'] = "reversed"
    
    fig.update_xaxes(showgrid=show_grid, gridcolor='#444')
    fig.update_yaxes(showgrid=show_grid, gridcolor='#444')

    st.plotly_chart(fig, use_container_width=True)
    
    # --- HEADER INSPECTOR ---
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.expander("View FITS Header"):
            # Handle dictionary vs FITS Header object
            header_dict = dict(header) if hasattr(header, 'items') else {k: str(v) for k, v in header.items()}
            st.table(header_dict)
    
    with col2:
        if is_demo:
            if st.button("❌ Exit Demo Mode", use_container_width=True):
                st.session_state.demo_mode = False
                st.rerun()

else:
    # --- LANDING PAGE (No File Loaded) ---
    st.info("Upload a FITS file above to begin analysis.")
    st.write("---")
    st.subheader("Practice with Demo Data")
    
    # Generate the demo data in memory (fast) so it's ready for both buttons
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
    
    # --- SEPARATE BUTTONS ---
    col1, col2 = st.columns(2)
    
    with col1:
        # Button 1: View in App
        if st.button("View Demo in App", use_container_width=True):
            st.session_state.demo_mode = True
            st.rerun()
            
    with col2:
        # Button 2: Download File
        st.download_button(
            label="Download FITS File",
            data=bio_gen,
            file_name="demo_spectrum.fits",
            mime="application/octet-stream",
            use_container_width=True
        )

