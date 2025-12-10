import numpy as np
import svgwrite
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.measure import find_contours
from PIL import Image
import argparse
import os


# ----------------------------
# IMAGE LOADING OR GENERATION
# ----------------------------
def load_or_generate_image(image_path=None, target_width=None):
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')

        if target_width:
            w_percent = (target_width / float(image.size[0]))
            h_size = int((float(image.size[1]) * float(w_percent)))
            image = image.resize((target_width, h_size), Image.LANCZOS)

        return np.array(image)
    else:
        # Synthetic fallback image, fixed size
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


# ----------------------------
# CONVERT CONTOUR TO SVG PATH
# ----------------------------
def contour_to_svg_path(contour, scale=1.0):
    path_str = "M " + " L ".join(f"{x*scale:.2f},{y*scale:.2f}" for y, x in contour)
    return path_str + " Z"


# ----------------------------
# BLOB ABSTRACTION WITH PERFECT EDGES
# ----------------------------
def abstract_tessellated_blobs(image, n_segments=40, compactness=10, output_path="tessellated_blobs.svg"):
    h, w, _ = image.shape
    image_float = img_as_float(image)
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=1)

    dwg = svgwrite.Drawing(filename=output_path, size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
    dwg.attribs['preserveAspectRatio'] = "xMidYMid meet"
    dwg.defs.add(dwg.style("path { stroke: none; stroke-width: 0; }"))

    unique_segments = np.unique(segments)

    for seg_id in unique_segments:
        mask = segments == seg_id
        avg_color = np.mean(image[mask], axis=0).astype(int)
        hex_color = svgwrite.rgb(*avg_color)

        contours = find_contours(mask.astype(float), level=0.5)

        for contour in contours:
            path_data = contour_to_svg_path(contour)
            dwg.add(dwg.path(d=path_data, fill=hex_color, stroke="none", stroke_width=0))

    dwg.save()
    print(f"[BLOB-TESSELLATED MODE] Saved to: {output_path}")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tessellated blob vectoriser for vinyl cutting.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image (optional)")
    parser.add_argument("--output", type=str, default="tessellated_blobs.svg", help="Output SVG path")
    parser.add_argument("--segments", type=int, default=40, help="Number of SLIC segments")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness")
    parser.add_argument("--resize", type=int, default=None, help="Resize image to target width (aspect ratio preserved)")

    args = parser.parse_args()
    image = load_or_generate_image(args.image, target_width=args.resize)
    abstract_tessellated_blobs(image, n_segments=args.segments, compactness=args.compactness, output_path=args.output)
