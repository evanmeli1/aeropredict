import streamlit as st
import numpy as np
import rasterio
import plotly.express as px
import folium
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from scipy.ndimage import gaussian_filter
from streamlit_folium import st_folium
from folium import plugins
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger("plotly").setLevel(logging.ERROR)


st.set_page_config(page_title="AeroPredict", layout="wide")
st.title("AeroPredict – Wildfire SAR Change Detection")

st.markdown("""
AeroPredict uses **Synthetic Aperture Radar (SAR)** data to analyze wildfire-affected regions.  
By comparing **VV** and **VH** polarizations, it detects surface and vegetation changes caused by fire events.  
Upload SAR GeoTIFFs from Sentinel-1 or NASA SAR missions and explore radar-based insights interactively.
""")

col1, col2 = st.columns(2)
file_vv = col1.file_uploader("Upload VV polarization (.tif)", type=["tif", "tiff"])
file_vh = col2.file_uploader("Upload VH polarization (.tif)", type=["tif", "tiff"])

MAX_SIZE = 512  # Downsample limit to avoid Streamlit size errors

def load_and_preprocess(file):
    with rasterio.open(file) as src:
        scale = max(src.width, src.height) / MAX_SIZE
        out_shape = (1, int(src.height / scale), int(src.width / scale))
        arr = src.read(1, out_shape=out_shape, resampling=Resampling.bilinear)
        arr = arr.astype("float32")
        arr[arr <= 0] = np.nan
        arr /= np.nanmax(arr)
        arr = gaussian_filter(arr, sigma=1)
        return arr, src.bounds, src.crs

if file_vv and file_vh:
    try:
        vv, bounds, crs = load_and_preprocess(file_vv)
        vh, _, _ = load_and_preprocess(file_vh)
        change = np.abs(vv - vh)

        st.header("SAR Polarizations and Change Detection")

        # Clean Plotly visuals (no warnings)
        fig_vv = px.imshow(vv, color_continuous_scale="Viridis", title="VV Polarization")
        fig_vh = px.imshow(vh, color_continuous_scale="Plasma", title="VH Polarization")
        fig_change = px.imshow(change, color_continuous_scale="Inferno", title="Change Detection Map")

        for fig in [fig_vv, fig_vh, fig_change]:
            fig.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                coloraxis_showscale=False,
            )

        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(fig_vv, config={"displayModeBar": False}, width="stretch")
        c2.plotly_chart(fig_vh, config={"displayModeBar": False}, width="stretch")
        c3.plotly_chart(fig_change, config={"displayModeBar": False}, width="stretch")

        # === Automated Insights ===
        st.subheader("Automated Insights & Hypothesis")
        high_change_ratio = np.nanmean(change > 0.5)
        if high_change_ratio > 0.2:
            st.write(f"🔥 Significant radar backscatter variation detected (≈{high_change_ratio*100:.1f}% of pixels).")
            st.write("This suggests widespread surface alteration, possibly vegetation loss or burn scars.")
        elif high_change_ratio > 0.05:
            st.write(f"⚠ Moderate change area detected (≈{high_change_ratio*100:.1f}% of pixels).")
            st.write("Local fire or partial vegetation disturbance likely present.")
        else:
            st.write(f"✅ Minimal change detected (≈{high_change_ratio*100:.1f}% of pixels).")
            st.write("No major wildfire impact apparent in the analyzed radar scene.")

        # === Geographic Map ===
        st.subheader("Geographic Context Map")
        if crs:
            lon_min, lat_min, lon_max, lat_max = transform_bounds(crs, "EPSG:4326", *bounds)
            fmap = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=9)
            folium.Rectangle(
                bounds=[(lat_min, lon_min), (lat_max, lon_max)],
                color="red",
                fill=True,
                fill_opacity=0.3
            ).add_to(fmap)
            plugins.MeasureControl().add_to(fmap)
            st_folium(fmap, height=450, width="100%")
        else:
            st.warning("This dataset has no CRS metadata, so a geographic overlay could not be created.")

        st.markdown("---")
        st.markdown("""
        ### About AeroPredict
        AeroPredict demonstrates how **SAR polarizations reveal post-fire surface changes** invisible to optical imagery.  
        - **VV polarization** detects horizontal scattering — useful for bare ground or water surfaces.  
        - **VH polarization** captures depolarized vertical scattering — sensitive to vegetation structure.  
        - The **change map** quantifies differences between these two to highlight burn severity.  
        
        This tool can be extended to flood zones, volcanic eruptions, or ice sheet monitoring — anywhere radar penetrates clouds and smoke.
        """)
        st.success("Analysis complete — ready for presentation.")

    except Exception as e:
        st.error(f"Could not process SAR data: {e}")

else:
    st.info("Upload both VV and VH GeoTIFFs to begin the analysis.")
