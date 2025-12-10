import numpy as np
import math
import os
import argparse
import svgwrite
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.util import img_as_float


# ----------------------------
# IMAGE LOADING OR GENERATION
# ----------------------------
def load_or_generate_image(image_path=None, size=(200, 200)):
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize(size)
        return np.array(image)
    else:
        # Generate synthetic gradient image
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        for i in range(size[0]):
            for j in range(size[1]):
                img[i, j] = [
                    int(127.5 + 127.5 * np.sin(i / 20)),
                    int(127.5 + 127.5 * np.cos(j / 30)),
                    int(127.5 + 127.5 * np.sin((i + j) / 40))
                ]
        return img


# ----------------------------
# SHAPE-BASED CELL ABSTRACTION
# ----------------------------
def abstract_cells(image, cell_size=20, n_colors=6, output_path="abstract_output_cells.svg"):
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape((h, w))
    palette = kmeans.cluster_centers_.astype(int)

    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x)  # radians

    dwg = svgwrite.Drawing(filename=output_path, size=(w, h))

    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            patch_label = labels[y, x]
            color = tuple(palette[patch_label])
            hex_color = svgwrite.rgb(color[0], color[1], color[2])
            cx, cy = x + cell_size // 2, y + cell_size // 2

            edge_mag = magnitude[y, x]
            edge_dir = direction[y, x]

            if edge_mag > 100:
                shape_type = "triangle"
            else:
                shape_type = "circle"

            if shape_type == "circle":
                dwg.add(dwg.circle(center=(cx, cy), r=cell_size // 3, fill=hex_color))
            elif shape_type == "triangle":
                points = [(-cell_size//2, cell_size//2), (cell_size//2, cell_size//2), (0, -cell_size//2)]
                rotated = []
                for px, py in points:
                    rx = px * math.cos(edge_dir) - py * math.sin(edge_dir)
                    ry = px * math.sin(edge_dir) + py * math.cos(edge_dir)
                    rotated.append((cx + rx, cy + ry))
                dwg.add(dwg.polygon(points=rotated, fill=hex_color))

    dwg.save()
    print(f"[CELL MODE] Saved to {output_path}")


# ----------------------------
# BLOB-BASED SLIC ABSTRACTION
# ----------------------------
def contour_to_svg_path(contour, scale=1.0):
    path_str = "M " + " L ".join(f"{x*scale:.2f},{y*scale:.2f}" for y, x in contour)
    return path_str + " Z"

def abstract_blobs(image, n_segments=40, compactness=10, output_path="abstract_output_blobs.svg"):
    h, w, _ = image.shape
    image_float = img_as_float(image)
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=1)
    

    dwg = svgwrite.Drawing(filename=output_path, size=(w, h))
    dwg.defs.add(dwg.style("path { stroke: none; stroke-width: 0; }"))
    unique_segments = np.unique(segments)

    for seg_id in unique_segments:
        mask = (segments == seg_id).astype(np.uint8)
        r, g, b = np.mean(image[segments == seg_id], axis=0).astype(int)
        hex_color = svgwrite.rgb(r, g, b)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] >= 3:
                path_data = contour_to_svg_path(contour)
                dwg.add(dwg.path(d=path_data, fill=hex_color, stroke="none", stroke_width=0))

    dwg.save()
    print(f"[BLOB MODE] Saved to {output_path}")


# ----------------------------
# MAIN ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-mode abstract vectoriser for vinyl-cuttable SVGs.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image (optional)")
    parser.add_argument("--output", type=str, default="abstract_output.svg", help="Output SVG path")
    parser.add_argument("--mode", type=str, choices=["cell", "blob"], default="cell", help="Abstraction mode: 'cell' or 'blob'")
    parser.add_argument("--cell_size", type=int, default=20, help="Grid cell size for cell mode")
    parser.add_argument("--colors", type=int, default=6, help="Number of dominant colours (cell mode)")
    parser.add_argument("--segments", type=int, default=40, help="Number of SLIC segments (blob mode)")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness (blob mode)")

    args = parser.parse_args()
    image = load_or_generate_image(args.image)

    if args.mode == "cell":
        abstract_cells(image, cell_size=args.cell_size, n_colors=args.colors, output_path=args.output)
    else:
        abstract_blobs(image, n_segments=args.segments, compactness=args.compactness, output_path=args.output)
