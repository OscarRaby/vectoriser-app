import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import svgwrite
import argparse
import os

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

def abstract_image_to_svg(image, cell_size=20, n_colors=6, output_path="abstract_output.svg"):
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape((h, w))
    palette = kmeans.cluster_centers_.astype(int)

    dwg = svgwrite.Drawing(filename=output_path, size=(w, h))
    
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            patch_label = labels[y, x]
            color = tuple(palette[patch_label])
            hex_color = svgwrite.rgb(color[0], color[1], color[2])
            cx, cy = x + cell_size // 2, y + cell_size // 2
            shape_type = np.random.choice(["circle", "rect", "triangle"])

            if shape_type == "circle":
                dwg.add(dwg.circle(center=(cx, cy), r=cell_size // 3, fill=hex_color))
            elif shape_type == "rect":
                dwg.add(dwg.rect(insert=(x, y), size=(cell_size, cell_size), fill=hex_color))
            elif shape_type == "triangle":
                points = [(x, y), (x + cell_size, y), (x + cell_size // 2, y + cell_size)]
                dwg.add(dwg.polygon(points=points, fill=hex_color))

    dwg.save()
    print(f"Saved abstract vector composition to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abstract vectorisation of an image for vinyl cutting.")
    parser.add_argument("--image", type=str, help="Path to input image (optional)", default=None)
    parser.add_argument("--output", type=str, help="Output SVG path", default="abstract_output.svg")
    parser.add_argument("--cell_size", type=int, default=20, help="Size of the abstraction grid cells")
    parser.add_argument("--colors", type=int, default=6, help="Number of dominant colours to use")

    args = parser.parse_args()
    img = load_or_generate_image(args.image)
    abstract_image_to_svg(img, cell_size=args.cell_size, n_colors=args.colors, output_path=args.output)