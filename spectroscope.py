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

# --- SIDEBAR UI (CONTROLS ONLY) ---
with st.sidebar:
    # 1. LOGO SECTION
    try:
        # Looks for the image in the main GitHub folder
        st.image("Nakshatra_transparent_1.png", use_container_width=True)
    except Exception:
        # Fallback if image is missing
        st.warning("⚠️ Logo 'Nakshatra_transparent_1.png' not found.")

    st.header("Nakshatra Club NITT") 
    st.caption("Telescope Team • Spectroscopy Division")
    st.divider()
    
    # 2. CALIBRATION (Keep in sidebar to save space)
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
    
    # 3. ANALYSIS TOOLS
    st.header("🔍 Analysis Tools")
    
    # Smoothing Slider
    smoothing_window = st.slider("Noise Reduction (Smoothing)", min_value=1, max_value=51, value=1, step=2)
    
    # Peak Finding
    show_peaks = st.checkbox("Auto-Detect Peaks", value=False)
    if show_peaks:
        peak_prominence = st.slider("Peak Sensitivity", 0.1, 50.0, 10.0)
    
    # Reference Lines
    st.subheader("Overlays")
    show_ref_lines = st.multiselect("Show Reference Lines", options=list(COMMON_LINES.keys()), default=[])

    st.divider()
    
    # 4. DISPLAY OPTIONS
    st.header("🎨 Display Options")
    show_grid = st.checkbox("Show Grid", value=True)
    normalize = st.checkbox("Normalize Intensity (0-1)", value=False)
    invert_yaxis = st.checkbox("Invert Y-Axis (Magnitudes)", value=False)

# --- MAIN PAGE HEADER ---
st.title("N-SIGHT (Nakshatra Spectroscopy Insight Generating High-tech Tool)")
st.markdown("**Telescope Team Project** | National Institute of Technology, Trichy")
st.markdown("---")

# --- DATA INPUT SECTION (MOVED TO MAIN SCREEN) ---
st.subheader("Data Input")
uploaded_file = st.file_uploader("Upload your FITS file here to begin analysis", type=["fit", "fits"])

# --- MAIN LOGIC ---

if uploaded_file is not None:
    try:
        # Open FITS file from memory
        with fits.open(uploaded_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            if data is None:
                st.error("No data found in Primary HDU.")
                st.stop()

            # --- DATA PROCESSING ---
            # Handle 2D Images
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

            # Apply Normalization
            if normalize:
                flux = normalize_data(flux)
                
            # Apply Smoothing (Savgol filter)
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

            # 1. Main Spectrum Line
            fig.add_trace(go.Scatter(
                x=x_axis, 
                y=flux, 
                mode='lines', 
                name='Spectrum Data',
                line=dict(color='#00CCFF', width=2)
            ))

            # 2. Detected Peaks
            if show_peaks and len(peak_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=x_axis[peak_indices],
                    y=flux[peak_indices],
                    mode='markers',
                    name='Detected Peaks',
                    marker=dict(color='red', size=8, symbol='x')
                ))
                for i in peak_indices:
                    fig.add_annotation(
                        x=x_axis[i], y=flux[i],
                        text=f"{x_axis[i]:.1f}Å",
                        showarrow=True, arrowhead=1, yshift=10
                    )

            # 3. Reference Lines
            for element_group in show_ref_lines:
                lines = COMMON_LINES[element_group]
                for name, wl in lines.items():
                    if x_axis.min() < wl < x_axis.max():
                        fig.add_vline(x=wl, line_width=1, line_dash="dash", line_color="yellow")
                        fig.add_annotation(
                            x=wl, y=0.95 if not invert_yaxis else 0.05, yref="paper",
                            text=name, showarrow=False, font=dict(color="yellow"), textangle=-90
                        )

            # Visual styling
            fig.update_layout(
                template="plotly_dark",
                xaxis_title=x_label,
                yaxis_title="Intensity",
                height=600,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            if invert_yaxis:
                fig['layout']['yaxis']['autorange'] = "reversed"
            
            fig.update_xaxes(showgrid=show_grid, gridcolor='#444')
            fig.update_yaxes(showgrid=show_grid, gridcolor='#444')

            st.plotly_chart(fig, use_container_width=True)
            
            # --- HEADER INSPECTOR ---
            with st.expander("View FITS Header"):
                header_dict = {k: str(v) for k, v in header.items()}
                st.table(header_dict)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    # Landing page content (Only shown when no file is uploaded)
    st.info("Upload a FITS file above to begin analysis.")
    st.write("Welcome to Nakshatra SpecLab. This tool allows the Telescope Team to analyze spectral data.")
    
    if st.button("Generate & Download Demo FITS"):
        x = np.linspace(4000, 7000, 2000) 
        continuum = 100 + (x - 4000) * 0.05
        h_alpha = 500 * np.exp(-0.5 * ((x - 6563) / 10)**2)
        h_beta = 300 * np.exp(-0.5 * ((x - 4861) / 10)**2)
        noise = np.random.normal(0, 5, 2000)
        y = continuum + h_alpha + h_beta + noise
        
        hdu = fits.PrimaryHDU(y)
        hdu.header['TELESCOP'] = 'Simulated Scope'
        bio = BytesIO()
        hdu.writeto(bio)
        bio.seek(0)
        
        st.download_button("Download demo_spectrum.fits", data=bio, file_name="demo_spectrum.fits")



