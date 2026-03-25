import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import streamlit as st

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

def compute_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def color_distance(img, target):
    diff = img.astype(float) - target.astype(float)
    return np.sqrt(np.sum(diff**2, axis=2))

def create_mask(rgb, target_rgb, tolerance):
    return color_distance(rgb, target_rgb) < tolerance

def apply_segmentation(rgb, active_indices, tolerance):
    h, w, _ = rgb.shape
    label_map = -1 * np.ones((h,w), dtype=int)

    for idx in active_indices:
        target = st.session_state.channel_targets[idx]
        mask = create_mask(rgb, target, tolerance)
        label_map[mask] = idx

    return label_map

def isolate_object(rgb, mask):
    out = np.ones_like(rgb) * 255
    out[mask] = rgb[mask]
    return out

# =====================================================
# DEFAULT COLORS
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

if "channel_targets" not in st.session_state:
    st.session_state.channel_targets = [None]*4

if "active_channel" not in st.session_state:
    st.session_state.active_channel = 0

if "spatial_results" not in st.session_state:
    st.session_state.spatial_results = {}

if "df_results" not in st.session_state:
    st.session_state.df_results = pd.DataFrame()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("Channels")

    for i in range(4):

        st.session_state.channels[i]["id"] = st.text_input(
            f"Channel {i+1} ID",
            st.session_state.channels[i]["id"],
            key=f"id_{i}"
        )

        st.session_state.channels[i]["color"] = st.selectbox(
            f"Color {i+1}",
            list(default_colors.keys()),
            index=list(default_colors.keys()).index(st.session_state.channels[i]["color"]),
            key=f"color_{i}"
        )

    selected = st.selectbox(
        "Active channel",
        [c["id"] for c in st.session_state.channels]
    )

    # =====================================================
    # ✅ FIX TARGET BEHAVIOR DI SINI (SATU-SATUNYA PERUBAHAN)
    # =====================================================
    if st.button("Set active"):
        new_idx = [
            c["id"] for c in st.session_state.channels
        ].index(selected)

        st.session_state.active_channel = new_idx

        # 🔥 reset target channel yang baru dipilih
        st.session_state.channel_targets[new_idx] = None

    # info aktif
    current_active = st.session_state.channels[st.session_state.active_channel]["id"]
    current_color = st.session_state.channels[st.session_state.active_channel]["color"]
    st.markdown(f"**Active: {current_active} ({current_color})**")

    tolerance = st.slider("Tolerance",5,120,40)
    alpha = st.slider("Overlay",0.1,1.0,0.5)

    # reset tetap ada
    if st.button("Reset targets"):
        st.session_state.channel_targets = [None]*4
        st.session_state.spatial_results = {}
        st.session_state.df_results = pd.DataFrame()

# =====================================================
# STEP 1
# =====================================================

st.header("Step 1: Define segmentation")

ref = st.file_uploader("Reference image")

if ref:

    img = Image.open(ref)
    rgb = pil_to_rgb(img)

    col1, col2 = st.columns(2)

    with col1:
        coords = streamlit_image_coordinates(img, key="ref")

    if coords:
        x, y = int(coords["x"]), int(coords["y"])

        patch = rgb[max(0,y-2):y+3, max(0,x-2):x+3]
        target = np.median(patch.reshape(-1,3), axis=0)

        st.session_state.channel_targets[st.session_state.active_channel] = target

    overlay = rgb.copy().astype(float)

    active_indices = [i for i,t in enumerate(st.session_state.channel_targets) if t is not None]

    for idx in active_indices:
        mask = create_mask(rgb, st.session_state.channel_targets[idx], tolerance)
        color = np.array(default_colors[st.session_state.channels[idx]["color"]])
        overlay[mask] = overlay[mask]*(1-alpha) + color*alpha

    with col2:
        st.image(overlay.astype(np.uint8), width="stretch")

# =====================================================
# STEP 2
# =====================================================

files = st.file_uploader("Upload images", accept_multiple_files=True)

# =====================================================
# RUN ANALYSIS
# =====================================================

if files and st.button("Run batch analysis"):

    active_indices = [i for i,t in enumerate(st.session_state.channel_targets) if t is not None]

    results = []
    spatial = {}

    for f in files:

        img = Image.open(f)
        rgb = pil_to_rgb(img)
        gray = compute_gray(rgb)

        label_map = apply_segmentation(rgb, active_indices, tolerance)

        spatial[f.name] = []

        total = np.sum(label_map >= 0)

        for idx in active_indices:

            ch = st.session_state.channels[idx]

            mask = label_map == idx
            area = np.sum(mask)

            percent = (area/total)*100 if total>0 else 0
            mean_intensity = float(np.mean(gray[mask])) if area>0 else 0

            results.append({
                "image": f.name,
                "label": ch["id"],
                "percent_area": percent,
                "mean_intensity": mean_intensity
            })

            spatial[f.name].append({
                "label": ch["id"],
                "image": isolate_object(rgb, mask),
                "percent": percent,
                "mean_intensity": mean_intensity
            })

    st.session_state.df_results = pd.DataFrame(results)
    st.session_state.spatial_results = spatial

# =====================================================
# VIEWER
# =====================================================

if st.session_state.spatial_results:

    st.subheader("Spatial Viewer")

    names = list(st.session_state.spatial_results.keys())
    idx = st.slider("Sample", 0, len(names)-1, 0)

    selected = names[idx]
    chs = st.session_state.spatial_results[selected]

    cols = st.columns(len(chs))

    for i, ch in enumerate(chs):
        with cols[i]:
            st.markdown(f"**{ch['label']}**")
            st.image(ch["image"], width="stretch")
            st.caption(f"{ch['percent']:.2f}% | {ch['mean_intensity']:.2f}")

# =====================================================
# TABLE
# =====================================================

if not st.session_state.df_results.empty:

    st.subheader("Results per Channel")

    df = st.session_state.df_results

    for label in df["label"].unique():
        st.markdown(f"### {label}")
        st.dataframe(df[df["label"] == label])

    st.download_button("Download CSV", df.to_csv(index=False), "results.csv")
