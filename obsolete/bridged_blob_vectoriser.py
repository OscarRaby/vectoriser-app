import numpy as np
import svgwrite
import argparse
import os
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.measure import find_contours
from scipy.spatial import KDTree
from skimage.color import rgb2lab


def load_or_generate_image(image_path=None, resize_width=None):
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        if resize_width:
            w_percent = resize_width / float(image.width)
            new_height = int(float(image.height) * w_percent)
            image = image.resize((resize_width, new_height), Image.LANCZOS)
        return np.array(image)
    else:
        # Synthetic fallback
        size = (200, 200)
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        for i in range(size[0]):
            for j in range(size[1]):
                img[i, j] = [
                    int(127.5 + 127.5 * np.sin(i / 20)),
                    int(127.5 + 127.5 * np.cos(j / 30)),
                    int(127.5 + 127.5 * np.sin((i + j) / 40))
                ]
        return img


def contour_to_svg_path(contour):
    path_str = "M " + " L ".join(f"{x:.2f},{y:.2f}" for y, x in contour)
    return path_str + " Z"


def bridge_contours(contours, centroids, colors, bridge_distance, color_tolerance, proximity_threshold):
    lab_colors = rgb2lab(np.array(colors).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    tree = KDTree(centroids)
    bridged_contours = []

    for i, contour in enumerate(contours):
        current_centroid = centroids[i]
        current_color = lab_colors[i]
        current_contour = np.copy(contour)

        indices = tree.query_ball_point(current_centroid, r=proximity_threshold)
        for j in indices:
            if j == i:
                continue
            color_diff = np.linalg.norm(current_color - lab_colors[j])
            if color_diff < color_tolerance:
                direction = np.array(centroids[j]) - np.array(current_centroid)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue
                direction = direction / norm
                distances = np.linalg.norm(current_contour - centroids[j], axis=1)
                idx = np.argmin(distances)
                current_contour[idx] = current_contour[idx] + direction * bridge_distance

        bridged_contours.append(current_contour)

    return bridged_contours


def create_bridged_svg(image, output_path, n_segments, compactness,
                       bridge_distance, color_tolerance, proximity_threshold):
    h, w, _ = image.shape
    image_float = img_as_float(image)
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=1)

    dwg = svgwrite.Drawing(filename=output_path, size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
    dwg.defs.add(dwg.style("path { stroke: none; stroke-width: 0; }"))

    unique_segments = np.unique(segments)
    contours, centroids, colors = [], [], []

    for seg_id in unique_segments:
        mask = segments == seg_id
        avg_color = np.mean(image[mask], axis=0).astype(int)
        found_contours = find_contours(mask.astype(float), level=0.5)
        if found_contours:
            contour = found_contours[0]
            contours.append(contour)
            centroids.append(np.mean(contour, axis=0))
            colors.append(avg_color)

    bridged = bridge_contours(contours, centroids, colors,
                              bridge_distance, color_tolerance, proximity_threshold)

    for contour, color in zip(bridged, colors):
        path_data = contour_to_svg_path(contour)
        hex_color = svgwrite.rgb(*color)
        dwg.add(dwg.path(d=path_data, fill=hex_color, stroke="none", stroke_width=0))

    dwg.save()
    print(f"[Bridged SVG] Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridged blob vectoriser for vinyl cutting.")
    parser.add_argument("--image", type=str, help="Path to input image (optional)", default=None)
    parser.add_argument("--output", type=str, help="Output SVG path", default="bridged_blobs.svg")
    parser.add_argument("--resize", type=int, help="Resize width for input image", default=None)
    parser.add_argument("--segments", type=int, default=40, help="Number of superpixels")
    parser.add_argument("--compactness", type=float, default=10.0, help="Compactness factor for SLIC")
    parser.add_argument("--bridge-distance", type=float, default=5.0, help="Distance to extend vertices")
    parser.add_argument("--color-tolerance", type=float, default=10.0, help="Color similarity threshold (LAB)")
    parser.add_argument("--proximity-threshold", type=float, default=50.0, help="Max distance to consider neighbour")

    args = parser.parse_args()
    img = load_or_generate_image(args.image, resize_width=args.resize)
    create_bridged_svg(
        img,
        output_path=args.output,
        n_segments=args.segments,
        compactness=args.compactness,
        bridge_distance=args.bridge_distance,
        color_tolerance=args.color_tolerance,
        proximity_threshold=args.proximity_threshold
    )
