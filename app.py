import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage import measure

import streamlit as st

# =========================
# DATA USER (hardcoded)
# =========================
USERS = {
    "admin": "12345",
    "lab": "iratco",
    "user1": "test"
}

# =========================
# SESSION INIT
# =========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# =========================
# LOGIN
# =========================
if not st.session_state.authenticated:
    col1, col2 = st.columns([8,2])
    with col1:
        st.title("🔐 Login to ")
    with col2:
        st.image("logo_iratco.png", width=250)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success(f"Selamat datang, {username} 👋")
        else:
            st.error("Username atau password salah")

    st.stop()

# =====================================================
# PAGE
# =====================================================
st.set_page_config(
    page_title="Image Segmentation Analysis",
    page_icon="logo.png",
    layout="wide"
)

# HEADER
col1, col2 = st.columns([8,2])
with col1:
    st.title("iRATco Image Segmentation Analysis")
    st.markdown("<span style='font-size:16px;color:gray;'>version 1.1.0</span>", unsafe_allow_html=True)
with col2:
    st.image("logo_iratco.png", width=250)

# =====================================================
# FUNCTIONS
# =====================================================

def pil_to_rgb(img):
    return np.array(img.convert("RGB"))

def color_distance(img, target):
    diff = img.astype(float) - target.astype(float)
    return np.sqrt(np.sum(diff**2, axis=2))

def create_mask(rgb, target_rgb, tolerance):
    return color_distance(rgb, target_rgb) < tolerance

def compute_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def density_plot(values):

    fig, ax = plt.subplots(figsize=(6,3))

    values = np.array(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        ax.text(0.5,0.5,"No data",ha="center")
        ax.set_axis_off()
        return fig

    ax.hist(values, bins=25, density=True, alpha=0.3)

    if len(values) > 1:
        xs = np.linspace(0,255,300)
        bw = max(np.std(values)*0.3, 1)

        density = np.zeros_like(xs)

        for v in values:
            density += np.exp(-0.5*((xs-v)/bw)**2)

        density /= (len(values)*bw*np.sqrt(2*np.pi))

        ax.plot(xs, density, linewidth=2)

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_xlim(0,255)

    return fig

# =====================================================
# COLOR OPTIONS
# =====================================================

default_colors = {
    "Red":[255,0,0],
    "Blue":[0,0,255],
    "Green":[0,255,0],
    "Brown":[165,42,42],
    "Yellow":[255,255,0],
    "White":[255,255,255]
}

# =====================================================
# SESSION STATE
# =====================================================

if "channels" not in st.session_state:
    st.session_state.channels = [
        {"id":"Channel 1","color":"Red"},
        {"id":"Channel 2","color":"Blue"},
        {"id":"Channel 3","color":"Green"},
        {"id":"Channel 4","color":"Brown"}
    ]

if "active_channel" not in st.session_state:
    st.session_state.active_channel = 0

if "label_map" not in st.session_state:
    st.session_state.label_map = None

if "last_click" not in st.session_state:
    st.session_state.last_click = None

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("Channel configuration")

    for i in range(4):

        st.session_state.channels[i]["id"] = st.text_input(
            f"ID {i+1}",
            value=st.session_state.channels[i]["id"],
            key=f"id_{i}"
        )

        st.session_state.channels[i]["color"] = st.selectbox(
            f"Color {i+1}",
            list(default_colors.keys()),
            index=list(default_colors.keys()).index(st.session_state.channels[i]["color"]),
            key=f"color_{i}"
        )

    st.markdown("---")

    selected_channel = st.selectbox(
        "Select active channel",
        [c["id"] for c in st.session_state.channels]
    )

    if st.button("Change label"):
        st.session_state.active_channel = [
            c["id"] for c in st.session_state.channels
        ].index(selected_channel)

    st.write("Active:", st.session_state.channels[st.session_state.active_channel]["id"])

    tolerance = st.slider("Color tolerance",5,120,40)
    alpha = st.slider("Overlay alpha",0.1,1.0,0.5)

    if st.button("Reset"):
        st.session_state.label_map = None
        st.session_state.last_click = None

# =====================================================
# UPLOAD
# =====================================================

uploaded = st.file_uploader("Upload IHC image", type=["png","jpg","jpeg","tif","tiff"])

if uploaded:

    image = Image.open(uploaded)
    rgb = pil_to_rgb(image)
    gray = compute_gray(rgb)

    h,w,_ = rgb.shape

    image_id = (h,w)
    if "image_id" not in st.session_state:
        st.session_state.image_id = image_id

    if st.session_state.image_id != image_id:
        st.session_state.label_map = None
        st.session_state.last_click = None
        st.session_state.image_id = image_id

    if st.session_state.label_map is None:
        st.session_state.label_map = -1 * np.ones((h,w), dtype=int)

    col1, col2 = st.columns(2)

    # CLICK
    with col1:
        st.subheader("RAW (click)")
        coords = streamlit_image_coordinates(image, key="img")

    if coords:
        click = (coords["x"], coords["y"])

        if click != st.session_state.last_click:

            st.session_state.last_click = click

            x = int(coords["x"])
            y = int(coords["y"])

            target = rgb[y,x]
            mask = create_mask(rgb, target, tolerance)

            idx = st.session_state.active_channel
            st.session_state.label_map[mask] = idx

    # PREVIEW
    overlay = rgb.copy().astype(float)

    for idx, ch in enumerate(st.session_state.channels):
        mask = st.session_state.label_map == idx

        if np.sum(mask)==0:
            continue

        color = np.array(default_colors[ch["color"]])
        overlay[mask] = overlay[mask]*(1-alpha) + color*alpha

    overlay = overlay.astype(np.uint8)

    with col2:
        st.subheader("Live preview")
        st.image(overlay, width="stretch")

    # =================================================
    # ANALYSIS (UPDATED WITH TOTAL OBJECT)
    # =================================================

    if st.button("Run analysis"):
        st.subheader("Segmentation Images Analysis")

        cols = st.columns(len(st.session_state.channels))
        total_area = np.sum(st.session_state.label_map >= 0)

        for idx, ch in enumerate(st.session_state.channels):

            mask = st.session_state.label_map == idx
            area = np.sum(mask)

            if area == 0:
                continue

            percent = (area / total_area) * 100
            mean_intensity = float(np.mean(gray[mask]))

            spatial = np.ones_like(rgb) * 255
            spatial[mask] = rgb[mask]

            with cols[idx]:

                st.markdown(f"### {ch['id']}")
                st.image(spatial, width="stretch")

                # ===== OBJECT SEGMENTATION =====
                labeled = measure.label(mask)
                props = measure.regionprops(labeled, intensity_image=gray)

                filtered_props = [p for p in props if p.area > 5]

                intensities = [p.mean_intensity for p in filtered_props]
                total_objects = len(filtered_props)

                # ===== METRICS =====
                st.markdown(
                    f"""
                    **Area:** {percent:.2f}%  
                    **Mean intensity:** {mean_intensity:.2f}  
                    **Total objects:** {total_objects}
                    """
                )

                # ===== DENSITY PLOT =====
                fig = density_plot(intensities)
                st.pyplot(fig)
                plt.close(fig)

else:
    st.info("Upload image to start")

st.markdown("---")

st.markdown("""
© 2026 Mawar Subangkit  
**iRATco Image Segmentation Analysis**  

If you use this software, please cite:

**Subangkit**, MAWAR (2026)  
**iRATco Image Segmentation Analysis**  
Available at: https://iratcosegment.streamlit.app/
""")
