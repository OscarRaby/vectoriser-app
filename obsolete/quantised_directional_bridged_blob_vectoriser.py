import numpy as np
import svgwrite
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.measure import find_contours
from scipy.spatial import KDTree
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

# -------------------------
# Synthetic Image Generator
# -------------------------
def generate_synthetic_image(size=(200, 200)):
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j] = [
                int(127.5 + 127.5 * np.sin(i / 20)),
                int(127.5 + 127.5 * np.cos(j / 30)),
                int(127.5 + 127.5 * np.sin((i + j) / 40))
            ]
    return img

# -------------------------
# Color Quantization
# -------------------------
def quantize_image_colors(image, max_colors):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=max_colors, n_init=10).fit(pixels)
    new_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
    quantized_image = new_pixels.reshape(image.shape)
    return quantized_image, kmeans.cluster_centers_

# -------------------------
# Convert Contour to SVG Path
# -------------------------
def contour_to_svg_path(contour, scale=1.0):
    path_str = "M " + " L ".join(f"{x*scale:.2f},{y*scale:.2f}" for y, x in contour)
    return path_str + " Z"

# -------------------------
# Bridging Algorithm
# -------------------------
def improved_bridge_contours(contours, centroids, colors,
                              bridge_distance=5.0,
                              color_tolerance=10.0,
                              proximity_threshold=50.0,
                              falloff_radius=5):
    lab_colors = rgb2lab(np.array(colors).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    tree = KDTree(centroids)
    bridged_contours = []

    for i, contour in enumerate(contours):
        current_color = lab_colors[i]
        current_centroid = centroids[i]
        current_contour = np.copy(contour)

        indices = tree.query_ball_point(current_centroid, r=proximity_threshold)
        for j in indices:
            if j == i:
                continue
            color_diff = np.linalg.norm(current_color - lab_colors[j])
            if color_diff < color_tolerance:
                distances = np.linalg.norm(current_contour - centroids[j], axis=1)
                idx = np.argmin(distances)
                target_contour = contours[j]
                closest_point = target_contour[np.argmin(
                    np.linalg.norm(target_contour - current_contour[idx], axis=1)
                )]

                direction = closest_point - current_contour[idx]
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue
                direction = direction / norm

                n = len(current_contour)
                for offset in range(-falloff_radius, falloff_radius + 1):
                    neighbor_idx = (idx + offset) % n
                    weight = 0.5 * (1 + np.cos(np.pi * offset / falloff_radius))
                    displacement = direction * bridge_distance * weight
                    current_contour[neighbor_idx] += displacement

        bridged_contours.append(current_contour)

    return bridged_contours

# -------------------------
# Main SVG Generation
# -------------------------
def create_directional_bridged_svg(image,
                                   output_path="directional_bridged_blobs_limited.svg",
                                   n_segments=40,
                                   compactness=10,
                                   max_colors=8,
                                   bridge_distance=5.0,
                                   color_tolerance=10.0,
                                   proximity_threshold=50.0,
                                   falloff_radius=5):
    h, w, _ = image.shape
    quantized_image, _ = quantize_image_colors(image, max_colors)
    image_float = img_as_float(quantized_image)
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=1)

    dwg = svgwrite.Drawing(filename=output_path, size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
    dwg.defs.add(dwg.style("path { stroke: none; stroke-width: 0; }"))

    unique_segments = np.unique(segments)
    contours, centroids, colors = [], [], []

    for seg_id in unique_segments:
        mask = segments == seg_id
        avg_color = np.mean(quantized_image[mask], axis=0).astype(int)
        found_contours = find_contours(mask.astype(float), level=0.5)
        if found_contours:
            contour = found_contours[0]
            contours.append(contour)
            centroids.append(np.mean(contour, axis=0))
            colors.append(avg_color)

    bridged = improved_bridge_contours(
        contours, centroids, colors,
        bridge_distance, color_tolerance,
        proximity_threshold, falloff_radius
    )

    for contour, color in zip(bridged, colors):
        hex_color = svgwrite.rgb(*color)
        path_data = contour_to_svg_path(contour)
        dwg.add(dwg.path(d=path_data, fill=hex_color))

    dwg.save()
    print(f"[SVG saved] {output_path}")
    return output_path

# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    image = generate_synthetic_image()
    create_directional_bridged_svg(
        image,
        output_path="directional_bridged_blobs_limited.svg",
        max_colors=8
    )
