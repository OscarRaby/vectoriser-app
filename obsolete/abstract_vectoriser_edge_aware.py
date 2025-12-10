import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import svgwrite
import cv2
import math
import argparse
import os
from skimage.segmentation import slic
from skimage.util import img_as_float
import cv2

def load_or_generate_image(image_path=None, size=(200, 200)):
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize(size)
        return np.array(image)
    else:
        # Generate synthetic image
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        for i in range(size[0]):
            for j in range(size[1]):
                r = int(127.5 + 127.5 * np.sin(i / 20))
                g = int(127.5 + 127.5 * np.cos(j / 30))
                b = int(127.5 + 127.5 * np.sin((i + j) / 40))
                img[i, j] = [r, g, b]
        return img

def contour_to_svg_path(contour, scale=1.0):
    path_str = "M " + " L ".join(f"{x*scale:.2f},{y*scale:.2f}" for y, x in contour)
    return path_str + " Z"

def abstract_image_to_svg(image, n_segments=40, compactness=10, output_path="abstract_output.svg"):
    h, w, _ = image.shape

    # Run SLIC segmentation
    image_float = img_as_float(image)
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=1)

    # Create SVG drawing
    dwg = svgwrite.Drawing(filename=output_path, size=(w, h))

    unique_segments = np.unique(segments)

    for seg_id in unique_segments:
        mask = (segments == seg_id).astype(np.uint8)

        # Calculate average colour of the region
        r, g, b = np.mean(image[segments == seg_id], axis=0).astype(int)
        hex_color = svgwrite.rgb(r, g, b)

        # Find external contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) == 2 and contour.shape[0] >= 3:
                path_data = contour_to_svg_path(contour)
                dwg.add(dwg.path(d=path_data, fill=hex_color))

    dwg.save()
    print(f"Saved abstract vector composition to: {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLIC-based abstract vectorisation.")
    parser.add_argument("--image", type=str, help="Path to input image (optional)", default=None)
    parser.add_argument("--output", type=str, help="Output SVG path", default="abstract_output.svg")
    parser.add_argument("--segments", type=int, default=40, help="Number of SLIC segments")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness factor")

    args = parser.parse_args()
    img = load_or_generate_image(args.image)
    abstract_image_to_svg(img, n_segments=args.segments, compactness=args.compactness, output_path=args.output)
