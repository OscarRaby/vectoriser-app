import streamlit as st
import numpy as np
from PIL import Image
from skimage import color
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage.measure import find_contours
import io
import svgwrite
import cairosvg
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from scipy.ndimage import label as nd_label, binary_dilation
from noise import pnoise2

# --- (Paste your functions here: quantize_image_colors, laplacian_smooth, improved_bridge_contours, etc.) ---

def quantize_image_colors(image, max_colors):
    if isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=max_colors, n_init=10).fit(pixels)
    new_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
    return new_pixels.reshape(image.shape), kmeans.cluster_centers_

def laplacian_smooth(contour, iterations=1, alpha=0.5):
    contour = np.copy(contour).astype(np.float32)
    n = len(contour)
    for _ in range(iterations):
        new_contour = np.copy(contour)
        for i in range(n):
            prev = contour[(i - 1) % n]
            next = contour[(i + 1) % n]
            new_contour[i] = (1 - alpha) * contour[i] + alpha * 0.5 * (prev + next)
        contour = new_contour
    return contour

def improved_bridge_contours(contours, centroids, colors, bridge_distance=5.0,
                              color_tolerance=10.0, proximity_threshold=50.0,
                              falloff_radius=5, max_curvature=np.radians(160)):
    lab_colors = rgb2lab(np.array(colors).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    tree = KDTree(centroids)
    bridged_contours = []

    for i, contour in enumerate(contours):
        current_color = lab_colors[i]
        current_centroid = centroids[i]
        current_contour = np.copy(contour)
        n = len(current_contour)

        indices = tree.query_ball_point(current_centroid, r=proximity_threshold)
        for j in indices:
            if j == i:
                continue
            color_diff = np.linalg.norm(current_color - lab_colors[j])
            if color_diff < color_tolerance:
                distances = np.linalg.norm(current_contour - centroids[j], axis=1)
                idx = np.argmin(distances)
                target_contour = contours[j]
                closest_point = target_contour[np.argmin(np.linalg.norm(target_contour - current_contour[idx], axis=1))]

                direction = closest_point - current_contour[idx]
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue
                direction = direction / norm

                for offset in range(-falloff_radius, falloff_radius + 1):
                    neighbor_idx = (idx + offset) % n
                    prev_idx = (neighbor_idx - 1) % n
                    next_idx = (neighbor_idx + 1) % n

                    v1 = current_contour[neighbor_idx] - current_contour[prev_idx]
                    v2 = current_contour[next_idx] - current_contour[neighbor_idx]
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    if norm_v1 == 0 or norm_v2 == 0:
                        continue
                    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                    if angle > max_curvature:
                        continue

                    weight = 0.5 * (1 + np.cos(np.pi * offset / falloff_radius))
                    displacement = direction * bridge_distance * weight
                    current_contour[neighbor_idx] += displacement

        bridged_contours.append(current_contour)

    return bridged_contours

def inflate_contour(contour, inflation_amount, far_point_factor=1.0):
    centroid = np.mean(contour, axis=0)
    vectors = contour - centroid
    distances = np.linalg.norm(vectors, axis=1)
    max_distance = np.max(distances)
    # Avoid division by zero
    norm_distances = distances / max_distance if max_distance > 0 else distances

    # Exponential inflation: base * exp(factor * normalized distance)
    inflation_factors = inflation_amount * np.exp((far_point_factor - 1) * norm_distances)
    norms = distances[:, np.newaxis]
    norms = np.where(norms == 0, 1, norms)
    directions = vectors / norms
    inflated = contour + directions * inflation_factors[:, np.newaxis]
    return inflated

def contour_to_svg_path(contour, scale=1.0):
    path_str = "M " + " L ".join(f"{x*scale:.2f},{y*scale:.2f}" for y, x in contour)
    return path_str + " Z"

# --- Segmentation Method ---
def noise_watershed(img, scale=60.0, blur_sigma=2.0, compactness=0.001):
    from skimage import filters, segmentation, morphology
    from scipy import ndimage as ndi
    from noise import pnoise2
    gray = color.rgb2gray(img)
    def perlin_noise(width, height, scale=10.0):
        return np.array([
            [pnoise2(x/scale, y/scale, octaves=3) for x in range(width)]
            for y in range(height)
        ])
    noise_field = perlin_noise(img.shape[1], img.shape[0], scale=scale)
    elevation_map = filters.gaussian(gray, sigma=blur_sigma) * 0.7 + noise_field * 0.3
    local_min = morphology.local_minima(elevation_map)
    markers, _ = ndi.label(local_min)
    labels = segmentation.watershed(elevation_map, markers=markers, compactness=compactness)
    return labels

def show_svg_as_png(svgstring, w, h):
    png_bytes = cairosvg.svg2png(bytestring=svgstring.encode("utf-8"), output_width=w, output_height=h)
    return Image.open(io.BytesIO(png_bytes))

st.set_page_config(page_title="Organic Segmentation Vectoriser", layout="wide")
st.title("Organic Segmentation Vectoriser (Streamlit Edition)")

# --- Image Upload ---
uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    pil_img = Image.open(uploaded_file).convert('RGB')
    # Downscale for speed
    pil_img = pil_img.resize((360, 240), Image.LANCZOS)
    img_np = img_as_float(np.array(pil_img))
    st.image(pil_img, caption="Input Image", width=360)
else:
    st.stop()

# --- Parameters ---
st.sidebar.header("Segmentation and Vectorisation Parameters")
noise_scale = st.sidebar.slider("Noise Scale", 10.0, 200.0, 60.0, step=1.0)
blur_sigma = st.sidebar.slider("Blur Sigma", 0.1, 10.0, 2.0, step=0.1)
compactness = st.sidebar.slider("Compactness", 0.0001, 0.01, 0.001, step=0.0001)
max_colors = st.sidebar.slider("Max Colors", 2, 32, 8, step=1)
bridge_distance = st.sidebar.slider("Bridge Distance", 0.0, 100.0, 5.0, step=1.0)
color_tolerance = st.sidebar.slider("Color Tolerance", 0.0, 100.0, 10.0, step=1.0)
proximity_threshold = st.sidebar.slider("Proximity Threshold", 0.0, 200.0, 50.0, step=1.0)
falloff_radius = st.sidebar.slider("Falloff Radius", 1, 30, 5, step=1)
max_curvature = st.sidebar.slider("Max Curvature", 1.0, 360.0, 160.0, step=1.0)
smooth_iterations = st.sidebar.slider("Smooth Iterations", 1, 20, 3, step=1)
smooth_alpha = st.sidebar.slider("Smooth Alpha", 0.01, 1.0, 0.3, step=0.01)
blob_inflation_amount = st.sidebar.slider("Blob Inflation Amount", 0.0, 50.0, 0.0, step=0.1)
far_point_inflation_factor = st.sidebar.slider("Far Point Inflation Factor", 0.0, 5.0, 1.0, step=0.01)
color_quantization = st.sidebar.checkbox("Enable Color Quantization", value=True)
bridging = st.sidebar.checkbox("Enable Bridging", value=True)
smoothing = st.sidebar.checkbox("Enable Smoothing", value=True)
inflation = st.sidebar.checkbox("Enable Inflation", value=True)

if st.button("Run Vectoriser"):
    # --- Noise Watershed Segmentation ---
    labels = noise_watershed(img_np, scale=noise_scale, blur_sigma=blur_sigma, compactness=compactness)

    # --- Color Quantization ---
    if color_quantization:
        quantized_image, _ = quantize_image_colors((img_np * 255).astype(np.uint8), max_colors)
    else:
        quantized_image = (img_np * 255).astype(np.uint8)

    # --- Find Contours & Colors ---
    unique_labels = np.unique(labels)
    contours, centroids, colors = [], [], []
    for seg_id in unique_labels:
        mask = (labels == seg_id)
        colors_in_mask = quantized_image[mask]
        if colors_in_mask.size == 0:
            continue
        unique_colors, counts = np.unique(colors_in_mask.reshape(-1, 3), axis=0, return_counts=True)
        main_color = tuple(unique_colors[np.argmax(counts)])
        found_contours = find_contours(mask.astype(float), level=0.5)
        if found_contours:
            contour = found_contours[0]
            contours.append(contour)
            centroids.append(np.mean(contour, axis=0))
            colors.append(main_color)

    # --- Bridging ---
    if bridging:
        bridged = improved_bridge_contours(
            contours, centroids, colors,
            bridge_distance=bridge_distance,
            color_tolerance=color_tolerance,
            proximity_threshold=proximity_threshold,
            falloff_radius=falloff_radius,
            max_curvature=np.radians(max_curvature)
        )
    else:
        bridged = contours

    # --- Smoothing ---
    if smoothing:
        bridged_smoothed = [
            laplacian_smooth(c, iterations=smooth_iterations, alpha=smooth_alpha)
            for c in bridged
        ]
    else:
        bridged_smoothed = bridged

    # --- Inflation ---
    inflation_scaled_contours = []
    for contour in bridged_smoothed:
        if inflation:
            inflated = inflate_contour(contour, blob_inflation_amount, far_point_inflation_factor)
        else:
            inflated = contour
        inflation_scaled_contours.append(inflated)

    # --- SVG Generation ---
    h, w = img_np.shape[:2]
    dwg = svgwrite.Drawing(size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
    dwg.defs.add(dwg.style("path { stroke: none; stroke-width: 0; }"))
    for contour, color_val in zip(inflation_scaled_contours, colors):
        hex_color = svgwrite.rgb(*color_val)
        path_data = contour_to_svg_path(contour)
        dwg.add(dwg.path(d=path_data, fill=hex_color))
    svg_string = dwg.tostring()

    # --- Show Output ---
    st.subheader("Output Preview")
    output_img = show_svg_as_png(svg_string, w, h)
    st.image(output_img, caption="Painterly Vectorised Output", width=360)
    st.download_button("Download SVG", svg_string, file_name="output.svg")

else:
    st.info("Adjust parameters and click **Run Vectoriser** to see the result.")