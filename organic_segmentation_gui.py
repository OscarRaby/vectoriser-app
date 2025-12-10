import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import color
from skimage.util import img_as_float
from PIL import Image
import io
import cairosvg
import svgwrite
from skimage.segmentation import slic
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from scipy.ndimage import label as nd_label, binary_dilation
from skimage.measure import find_contours, approximate_polygon
from io import BytesIO

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
            prev_pt = contour[(i - 1) % n]
            next_pt = contour[(i + 1) % n]
            new_contour[i] = (1 - alpha) * contour[i] + alpha * 0.5 * (prev_pt + next_pt)
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

def contour_to_svg_path(contour, scale=1.0, simplify_tol=0.1, quantize=True):
    """
    Convert contour (array of [y,x] points) to an SVG path string.
    - simplify_tol: tolerance (pixels) for approximate_polygon (0 = no simplification)
    - quantize: reduce coordinate precision to integers to shrink output
    """
    # contour is [N,2] as (row=y, col=x)
    pts = np.asarray(contour)
    if pts.shape[0] == 0:
        return ""
    # approximate_polygon expects coordinates as (row,col)
    if simplify_tol and simplify_tol > 0:
        try:
            pts = approximate_polygon(pts, tolerance=simplify_tol)
        except Exception:
            # fallback: keep original if approximation fails
            pts = np.asarray(contour)

    if quantize:
        xs = np.round(pts[:,1] * scale).astype(int)
        ys = np.round(pts[:,0] * scale).astype(int)
        coord_pairs = [f"{x},{y}" for x, y in zip(xs, ys)]
    else:
        coord_pairs = [f"{x*scale:.1f},{y*scale:.1f}" for y, x in pts]

    if not coord_pairs:
        return ""
    path_str = "M " + " L ".join(coord_pairs) + " Z"
    return path_str

def generate_droplets(
    contour, direction, num_droplets=5,
    min_dist=10, max_dist=30,
    size_mean=5, size_std=2,
    spread_angle=np.radians(10),
    simplify_tol=0.5
):
    """Generate polygonal approximations of painterly droplets.

    Returns a list of numpy arrays (Npoints, 2) in (row, col) ordering suitable for contour_to_svg_path.
    """
    import math
    droplets = []
    try:
        contour = np.asarray(contour)
        if contour.size == 0 or num_droplets <= 0:
            return []
    except Exception:
        return []

    mean_pt = np.mean(contour, axis=0)
    base_angle = math.atan2(direction[0], direction[1]) if hasattr(direction, '__len__') else 0.0
    for i in range(int(max(0, num_droplets))):
        dist = float(np.random.uniform(min_dist, max_dist))
        ang = base_angle + float(np.random.uniform(-spread_angle/2, spread_angle/2))
        cy = mean_pt[0] + math.sin(ang) * dist
        cx = mean_pt[1] + math.cos(ang) * dist
        rx = float(abs(np.random.normal(size_mean, size_std)))
        ry = float(abs(np.random.normal(size_mean * 0.8, max(size_std*0.8, 0.1))))
        rx = max(0.5, rx)
        ry = max(0.5, ry)
        # polygon approximation of an ellipse
        n_pts = 14
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        ys = cy + (np.sin(angles) * ry * np.cos(ang) - np.cos(angles) * rx * np.sin(ang))
        xs = cx + (np.cos(angles) * rx * np.cos(ang) + np.sin(angles) * ry * np.sin(ang))
        poly = np.column_stack([ys, xs])
        # simplify if requested
        if simplify_tol and simplify_tol > 0:
            try:
                poly_s = np.asarray(approximate_polygon(poly, tolerance=simplify_tol))
                if poly_s.size > 0:
                    poly = poly_s
            except Exception:
                pass
        droplets.append(poly)
    return droplets


def generate_painterly_ellipses(
    contour, direction, num_droplets=5,
    min_dist=10, max_dist=30,
    size_mean=5, size_std=2,
    spread_angle=np.radians(10),
    rotation_offset=0.0
):
    """Generate lightweight ellipse descriptors suitable for emission as native SVG <ellipse> elements.

    Returns a list of dicts: {'type':'ellipse','cx':..., 'cy':..., 'rx':..., 'ry':..., 'angle': ...}
    Coordinates are in SVG space (cx = column, cy = row).
    """
    import math
    ellipses = []
    try:
        contour = np.asarray(contour)
        if contour.size == 0 or num_droplets <= 0:
            return []
    except Exception:
        return []

    mean_pt = np.mean(contour, axis=0)
    base_angle = math.atan2(direction[0], direction[1]) if hasattr(direction, '__len__') else 0.0
    for i in range(int(max(0, num_droplets))):
        dist = float(np.random.uniform(min_dist, max_dist))
        ang = base_angle + float(np.random.uniform(-spread_angle/2, spread_angle/2))
        cy = mean_pt[0] + math.sin(ang) * dist
        cx = mean_pt[1] + math.cos(ang) * dist
        rx = float(abs(np.random.normal(size_mean, size_std)))
        ry = float(abs(np.random.normal(size_mean * 0.8, max(size_std*0.8, 0.1))))
        rx = max(0.5, rx)
        ry = max(0.5, ry)
        angle_deg = float(np.degrees(ang)) + float(rotation_offset)
        ellipses.append({'type': 'ellipse', 'cx': float(cx), 'cy': float(cy), 'rx': float(rx), 'ry': float(ry), 'angle': angle_deg})
    return ellipses

def generate_painterly_rects(
    contour, direction, num_droplets=5,
    min_dist=10, max_dist=30,
    size_mean=5, size_std=2,
    spread_angle=np.radians(10),
    rotation_offset=0.0, horizontal=False
):
    """Generate lightweight rectangle descriptors suitable for emission as native SVG <rect> elements.

    Returns a list of dicts: {'type':'rect','cx':..., 'cy':..., 'w':..., 'h':..., 'angle': ...}
    Coordinates are in SVG space (cx = column, cy = row).
    """
    import math
    rects = []
    try:
        contour = np.asarray(contour)
        if contour.size == 0 or num_droplets <= 0:
            return []
    except Exception:
        return []

    mean_pt = np.mean(contour, axis=0)
    base_angle = math.atan2(direction[0], direction[1]) if hasattr(direction, '__len__') else 0.0
    for i in range(int(max(0, num_droplets))):
        dist = float(np.random.uniform(min_dist, max_dist))
        ang = base_angle + float(np.random.uniform(-spread_angle/2, spread_angle/2))
        cy = mean_pt[0] + math.sin(ang) * dist
        cx = mean_pt[1] + math.cos(ang) * dist
        w_rect = float(abs(np.random.normal(size_mean * 2.0, size_std)))
        h_rect = float(abs(np.random.normal(size_mean * 0.8, max(size_std*0.5, 0.1))))
        w_rect = max(0.5, w_rect)
        h_rect = max(0.5, h_rect)
        angle_deg = float(np.degrees(ang)) + float(rotation_offset)
        # If horizontal flag is set, swap width/height so rectangles are elongated horizontally
        if horizontal:
            w_use, h_use = max(w_rect, h_rect), min(w_rect, h_rect)
        else:
            w_use, h_use = w_rect, h_rect
        rects.append({'type': 'rect', 'cx': float(cx), 'cy': float(cy), 'w': float(w_use), 'h': float(h_use), 'angle': angle_deg})
    return rects


def generate_organic_droplets(
    contour, direction, num_droplets=5,
    min_dist=10, max_dist=30,
    size_mean=5, size_std=2,
    spread_angle=np.radians(10),
    organic_strength=1.0,
    jitter_amount=0.5,
    elongation=0.0,
    simplify_tol=0.5
):
    """Generate organic, irregular droplet polygons using Perlin-like noise when available.

    Falls back to smooth random modulation if the `noise` package is not installed.
    """
    try:
        from noise import pnoise1
    except Exception:
        pnoise1 = None
    droplets = []
    direction = np.array(direction, dtype=float)
    if np.linalg.norm(direction) == 0:
        direction = np.array([1.0, 0.0])
    direction = direction / np.linalg.norm(direction)
    theta = np.linspace(0, 2*np.pi, 12)
    for _ in range(max(0, int(num_droplets))):
        start_point = contour[np.random.randint(len(contour))]
        dist = float(np.random.uniform(min_dist, max_dist))
        angle_jitter = float(np.random.uniform(-spread_angle, spread_angle))
        c, s = np.cos(angle_jitter), np.sin(angle_jitter)
        dir_rot = np.array([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]])
        norm_dir_rot = np.linalg.norm(dir_rot)
        if norm_dir_rot == 0:
            dir_rot = direction.copy()
        else:
            dir_rot = dir_rot / norm_dir_rot
        ortho = np.array([-dir_rot[1], dir_rot[0]])
        norm_ortho = np.linalg.norm(ortho)
        if norm_ortho == 0:
            ortho = np.array([0.0, 1.0])
        else:
            ortho = ortho / norm_ortho

        center = start_point + dist * dir_rot
        size = np.abs(np.random.normal(size_mean, size_std, 2))
        if elongation > 0:
            size_major = size[0] * (1.0 + elongation)
            size_minor = size[1] / (1.0 + elongation)
        elif elongation < 0:
            factor = max(0.01, 1.0 + elongation)
            size_major = size[0] * factor
            size_minor = size[1] / factor
        else:
            size_major = size[0]
            size_minor = size[1]

        # Organic radius modulation using Perlin noise or random fallback
        if pnoise1 is not None:
            radii = size_major + (np.array([pnoise1(t * 2 + np.random.rand()*10) for t in theta]) * organic_strength)
        else:
            radii = size_major + (np.random.randn(len(theta)) * 0.3 * organic_strength)
        xs = center[0] + radii * np.cos(theta) * dir_rot[0] + size_minor * np.sin(theta) * ortho[0]
        ys = center[1] + radii * np.cos(theta) * dir_rot[1] + size_minor * np.sin(theta) * ortho[1]
        ellipse = np.stack([xs, ys], axis=-1)
        ellipse += np.random.normal(0, jitter_amount, ellipse.shape)

        try:
            simple = approximate_polygon(ellipse, tolerance=simplify_tol)
            if len(simple) >= 3:
                droplets.append(simple)
            else:
                droplets.append(ellipse)
        except Exception:
            droplets.append(ellipse)
    return droplets


# Robust noise-based watershed segmentation (keeps previous behavior but guarded)
def noise_watershed(img, scale=60.0, blur_sigma=2.0, compactness=0.001):
    from skimage import filters, segmentation, morphology
    from scipy import ndimage as ndi
    try:
        from noise import pnoise2
    except Exception:
        pnoise2 = None
    gray = color.rgb2gray(img)

    def perlin_noise(width, height, scale=10.0):
        if pnoise2 is None:
            return np.zeros((height, width), dtype=float)
        return np.array([
            [pnoise2(x/scale, y/scale, octaves=3) for x in range(width)]
            for y in range(height)
        ])

    noise_field = perlin_noise(img.shape[1], img.shape[0], scale=scale)
    elevation_map = filters.gaussian(gray, sigma=blur_sigma) * 0.7 + noise_field * 0.3
    local_min = morphology.local_minima(elevation_map)
    lab_res = ndi.label(local_min)
    if isinstance(lab_res, tuple):
        markers = lab_res[0]
    else:
        markers = lab_res
    labels = segmentation.watershed(elevation_map, markers=markers, compactness=int(round(compactness)))
    return labels

class OrganicSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Organic Segmentation Vectoriser")
        self.image = None
        self.img_np = None
        self.labels = None
        self.svg_path = tk.StringVar(value="out.svg")
        # --- Parameters ---
        self.param_vars = {
            "Noise Scale": tk.DoubleVar(value=60.0),
            "Blur Sigma": tk.DoubleVar(value=2.0),
            "Compactness": tk.DoubleVar(value=0.001),
            "Max Colors": tk.IntVar(value=8),
            "Bridge Distance": tk.DoubleVar(value=5.0),
            "Color Tolerance": tk.DoubleVar(value=10.0),
            "Proximity Threshold": tk.DoubleVar(value=50.0),
            "Falloff Radius": tk.IntVar(value=5),
            "Max Curvature": tk.DoubleVar(value=160.0),
            "Smooth Iterations": tk.IntVar(value=3),
            "Smooth Alpha": tk.DoubleVar(value=0.3),
            "Blob Inflation Amount": tk.DoubleVar(value=0.0),
            "Far Point Inflation Factor": tk.DoubleVar(value=1.0),
            "Inflation Proportional to Stacking": tk.BooleanVar(value=True),
            "Stacking Order": tk.StringVar(value="area"),
            "Segmentation Multiplier": tk.DoubleVar(value=1.0),
            # --- Splatter/Splatter Droplet Parameters ---
            "Droplet Density": tk.IntVar(value=0),
            "Droplet Min Distance": tk.DoubleVar(value=5.0),
            "Droplet Max Distance": tk.DoubleVar(value=15.0),
            "Droplet Size Mean": tk.DoubleVar(value=3.0),
            "Droplet Size Std": tk.DoubleVar(value=1.0),
            "Droplet Spread": tk.DoubleVar(value=5.0),
            # --- Organic droplet controls ---
            "Droplet Organic Min Brightness": tk.DoubleVar(value=128.0),
            "Droplet Organic Density": tk.IntVar(value=3),
            "Droplet Organic Strength": tk.DoubleVar(value=1.0),
            "Droplet Organic Jitter": tk.DoubleVar(value=0.5),
            "Droplet Organic Elongation": tk.DoubleVar(value=0.0),
            # New: Organic droplet percent per blob
            "Droplet Organic Percent Per Blob": tk.DoubleVar(value=100.0),
            # Option: render painterly droplets as native lightweight SVG <ellipse> elements
            "Painterly Use SVG Ellipses": tk.BooleanVar(value=False),
            # Global rotation offset (degrees) applied to painterly primitives
            "Droplet Global Rotation": tk.DoubleVar(value=0.0),
            # If True, generated painterly rects will be wider horizontally (swap width/height)
            "Painterly Rect Horizontal": tk.BooleanVar(value=False),
            # Contour/droplet simplification tolerance (pixels)
            "Simplify Tolerance": tk.DoubleVar(value=0.5)
        }
        # Option: enable/disable drawing a native vector preview into the main axes
        # (useful to turn off if cairosvg rasterization is preferred or slow)
        self.param_vars["Enable Vector Preview"] = tk.BooleanVar(value=True)
        # --- GUI layout ---
        # Use a scrollable frame for the left controls
        from tkinter import Canvas, Frame, Scrollbar
        # Main frame with 3 columns: left (controls), middle (buttons/preview), right (matplotlib)
        self.main_frame = Frame(root)
        self.main_frame.grid(row=0, column=0, sticky='nsew')
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=1)

        # --- Column 0: scrollable controls ---
        self.canvas = Canvas(self.main_frame, borderwidth=0, background="#f0f0f0", width=400)
        self.scroll_frame = Frame(self.canvas, background="#f0f0f0")
        self.vsb = Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        # Use grid instead of pack for canvas and scrollbar
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.canvas.create_window((0,0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # Configure grid weights for resizing
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=0)
        self.main_frame.grid_columnconfigure(2, weight=1)
        # Now use self.scroll_frame for all left-side widgets
        left_root = self.scroll_frame

        # --- Column 1: buttons and preview controls ---
        self.button_frame = Frame(self.main_frame)
        # Place the button/preview column in the middle (column 1)
        self.button_frame.grid(row=0, column=1, sticky='ns')
        
        # --- Column 2: matplotlib output ---
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        # Place the matplotlib output in the rightmost column (column 2)
        self.canvas_plot.get_tk_widget().grid(row=0, column=2, rowspan=999, sticky='nsew')
        self.root.grid_rowconfigure(0, weight=1)
        # Ensure column 2 (right) expands
        self.root.grid_columnconfigure(2, weight=1)

        # Now place all control widgets in left_root, and all buttons/preview in self.button_frame
        root = left_root

        tk.Button(root, text="Load Image", command=self.load_image).grid(row=0, column=0, columnspan=2, pady=5)
        row = 1
        for key in [
            "Noise Scale", "Blur Sigma", "Compactness", "Max Colors", "Bridge Distance", "Color Tolerance",
            "Proximity Threshold", "Falloff Radius", "Max Curvature", "Smooth Iterations", "Smooth Alpha",
            "Blob Inflation Amount", "Far Point Inflation Factor",
            # Add splatter parameters to GUI
            "Droplet Density", "Droplet Min Distance", "Droplet Max Distance", "Droplet Size Mean", "Droplet Size Std", "Droplet Spread",
            # --- Organic droplet controls ---
            "Droplet Organic Min Brightness", "Droplet Organic Density", "Droplet Organic Strength", "Droplet Organic Jitter", "Droplet Organic Elongation",
            # New: add to GUI
            "Droplet Organic Percent Per Blob",
            # Simplification tolerance for contours and droplets
            "Simplify Tolerance"
        ]:
            if key == "Simplify Tolerance":
                tk.Label(root, text=key+" (px)").grid(row=row, column=0, sticky='e')
                # ensure we reuse existing var if present
                if key not in self.param_vars:
                    self.param_vars[key] = tk.DoubleVar(value=0.5)
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=5.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=row, column=1)
                row += 1
                continue
            if key == "Droplet Organic Min Brightness":
                tk.Label(root, text=key+" (0-255, organic)").grid(row=row, column=0, sticky='e')
                if key not in self.param_vars:
                    self.param_vars[key] = tk.DoubleVar(value=128.0)
                tk.Scale(root, variable=self.param_vars[key], from_=0, to=255, resolution=1, orient=tk.HORIZONTAL).grid(row=row, column=1)
                row += 1
                continue
            if key == "Droplet Organic Density":
                tk.Label(root, text=key+" (organic)").grid(row=row, column=0, sticky='e')
                if key not in self.param_vars:
                    self.param_vars[key] = tk.IntVar(value=3)
                tk.Scale(root, variable=self.param_vars[key], from_=0, to=20, resolution=1, orient=tk.HORIZONTAL).grid(row=row, column=1)
                row += 1
                continue
            if key == "Droplet Organic Strength":
                tk.Label(root, text=key+" (organic)").grid(row=row, column=0, sticky='e')
                if key not in self.param_vars:
                    self.param_vars[key] = tk.DoubleVar(value=1.0)
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=5.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=row, column=1)
                row += 1
                continue
            if key == "Droplet Organic Jitter":
                tk.Label(root, text=key+" (organic)").grid(row=row, column=0, sticky='e')
                if key not in self.param_vars:
                    self.param_vars[key] = tk.DoubleVar(value=0.5)
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=5.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=row, column=1)
                row += 1
                continue
            if key == "Droplet Organic Elongation":
                tk.Label(root, text=key+" (organic)").grid(row=row, column=0, sticky='e')
                if key not in self.param_vars:
                    self.param_vars[key] = tk.DoubleVar(value=0.0)
                tk.Scale(root, variable=self.param_vars[key], from_=-2.0, to=5.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=row, column=1)
                row += 1
                continue
            if key == "Droplet Organic Percent Per Blob":
                tk.Label(root, text=key+" (%)").grid(row=row, column=0, sticky='e')
                if key not in self.param_vars:
                    self.param_vars[key] = tk.DoubleVar(value=100.0)
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=100.0, resolution=0.1, orient=tk.HORIZONTAL).grid(row=row, column=1)
                row += 1
                continue
            tk.Label(root, text=key).grid(row=row, column=0, sticky='e')
            # Use sliders for parameter entry
            if key == "Noise Scale":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=1000.0, resolution=10.0, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Blur Sigma":
                tk.Scale(root, variable=self.param_vars[key], from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Compactness":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0001, to=1, resolution=0.001, orient=tk.HORIZONTAL).grid(row=row, column=1)  # Avoid zero
            elif key == "Max Colors":
                tk.Scale(root, variable=self.param_vars[key], from_=2, to=32, resolution=1, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Bridge Distance":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=100.0, resolution=1.0, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Color Tolerance":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=100.0, resolution=1.0, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Proximity Threshold":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=200.0, resolution=1.0, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Falloff Radius":
                tk.Scale(root, variable=self.param_vars[key], from_=1, to=30, resolution=1, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Max Curvature":
                tk.Scale(root, variable=self.param_vars[key], from_=1.0, to=360.0, resolution=1.0, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Smooth Iterations":
                tk.Scale(root, variable=self.param_vars[key], from_=1, to=20, resolution=1, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Smooth Alpha":
                tk.Scale(root, variable=self.param_vars[key], from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=row, column=1)  # Avoid zero
            elif key == "Blob Inflation Amount":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=50.0, resolution=0.1, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Far Point Inflation Factor":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=5.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Droplet Density":
                tk.Scale(root, variable=self.param_vars[key], from_=0, to=10, resolution=1, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Droplet Min Distance":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=100.0, resolution=1.0, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Droplet Max Distance":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=200.0, resolution=1.0, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Droplet Size Mean":
                tk.Scale(root, variable=self.param_vars[key], from_=1.0, to=50.0, resolution=0.1, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Droplet Size Std":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Droplet Spread":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0, to=90.0, resolution=1.0, orient=tk.HORIZONTAL).grid(row=row, column=1)
            row += 1
        tk.Label(root, text="Inflation Proportional to Stacking").grid(row=row, column=0, sticky='e')
        tk.Checkbutton(root, variable=self.param_vars["Inflation Proportional to Stacking"]).grid(row=row, column=1)
        row += 1
        tk.Label(root, text="Stacking Order").grid(row=row, column=0, sticky='e')
        stacking_options = ["area", "area_reverse", "brightness", "brightness_reverse", "position_x", "position_x_reverse", "position_y", "position_y_reverse", "position_centre", "position_centre_reverse"]
        tk.OptionMenu(root, self.param_vars["Stacking Order"], *stacking_options).grid(row=row, column=1)
        row += 1
        tk.Label(root, text="SVG Output Path").grid(row=row, column=0, sticky='e')
        tk.Entry(root, textvariable=self.svg_path).grid(row=row, column=1)
        row += 1
        # --- Move all widgets below here to button_frame (column 2) ---
        # Modifier switches
        self.modifier_vars = {
            "Color Quantization": tk.BooleanVar(value=True),
            "Bridging": tk.BooleanVar(value=True),
            "Smoothing": tk.BooleanVar(value=True),
            "Inflation": tk.BooleanVar(value=True),
        }
        button_row = 0
        self.run_button = tk.Button(self.button_frame, text="Run Vectoriser", command=self.run_vectoriser)
        self.run_button.grid(row=button_row, column=0, columnspan=2, pady=5)
        button_row += 1
        for mod_key in self.modifier_vars:
            tk.Checkbutton(self.button_frame, text=f"Enable {mod_key}", variable=self.modifier_vars[mod_key]).grid(row=button_row, column=0, sticky='w')
            button_row += 1
        # Scaling method selector for segmentation parameters
        self.scaling_method = tk.StringVar(value="max")
        tk.Label(self.button_frame, text="Segmentation Scaling").grid(row=button_row, column=0, sticky='e')
        tk.OptionMenu(self.button_frame, self.scaling_method, "max", "min", "average", "area", "sqrt_area").grid(row=button_row, column=1)
        button_row += 1
        tk.Label(self.button_frame, text="Segmentation Parameter Divider").grid(row=button_row, column=0, sticky='e')
        tk.Scale(self.button_frame, variable=self.param_vars["Segmentation Multiplier"], from_=0.1, to=10.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=button_row, column=1)
        button_row += 1
        # Progress bar (move below Run button)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = tk.Scale(self.button_frame, variable=self.progress_var, from_=0, to=100, orient=tk.HORIZONTAL, length=300, showvalue=False)
        self.progress_bar.grid(row=button_row, column=0, columnspan=2, pady=5)
        button_row += 1
        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = tk.Label(self.button_frame, textvariable=self.status_var, anchor='w', relief=tk.SUNKEN, bd=1)
        self.status_label.grid(row=button_row, column=0, columnspan=2, sticky='we')
        button_row +=1

        # Add droplet style toggle to button_frame
        self.droplet_style = tk.StringVar(value="painterly")
        # Keep label and option on the same row to avoid overlap
        tk.Label(self.button_frame, text="Droplet Style").grid(row=button_row, column=0, sticky='e')
        tk.OptionMenu(self.button_frame, self.droplet_style, "painterly", "organic").grid(row=button_row, column=1)
        button_row += 1
        # Checkbox to control whether painterly droplets are emitted as native SVG primitives (ellipse or rect)
        tk.Checkbutton(self.button_frame, text="Use SVG primitives for painterly droplets", variable=self.param_vars["Painterly Use SVG Ellipses"]).grid(row=button_row, column=0, columnspan=2, sticky='w')
        button_row += 1
        # Global rotation applied to painterly primitives (degrees)
        tk.Label(self.button_frame, text="Global Droplet Rotation").grid(row=button_row, column=0, sticky='e')
        tk.Scale(self.button_frame, variable=self.param_vars["Droplet Global Rotation"], from_=-180, to=180, resolution=1, orient=tk.HORIZONTAL).grid(row=button_row, column=1)
        button_row += 1
        # Option to force painterly rects to be horizontally elongated
        tk.Checkbutton(self.button_frame, text="Rect Horizontal (w>h)", variable=self.param_vars["Painterly Rect Horizontal"]).grid(row=button_row, column=0, columnspan=2, sticky='w')
        button_row += 1
        # Option: choose primitive type when emitting painterly SVG droplets
        self.param_vars["Painterly SVG Primitive"] = tk.StringVar(value="ellipse")
        tk.Label(self.button_frame, text="Painterly SVG Primitive").grid(row=button_row, column=0, sticky='e')
        tk.OptionMenu(self.button_frame, self.param_vars["Painterly SVG Primitive"], "ellipse", "rect").grid(row=button_row, column=1, columnspan=1, sticky='w')
        button_row += 1
        # Checkbox to enable/disable drawing the live vector preview into the main axes
        tk.Checkbutton(self.button_frame, text="Enable Vector Preview (main view)", variable=self.param_vars["Enable Vector Preview"]).grid(row=button_row, column=0, columnspan=2, sticky='w')
        button_row += 1

        # Preview controls (manual only). No thumbnail in middle column to keep layout compact.
        preview_row = button_row
        tk.Button(self.button_frame, text="Preview Parameters", command=self.run_preview).grid(row=preview_row, column=0, columnspan=3, sticky='ew', pady=2)
        preview_row += 1

        # Preset controls
        self.preset_name_var = tk.StringVar()
        self.preset_menu_var = tk.StringVar()
        self.presets_file = 'vectoriser_presets.json'
        import json
        try:
            with open(self.presets_file, 'r') as f:
                self.presets = json.load(f)
        except Exception:
            self.presets = {}
        self.PRESET_PARAMS = [k for k in self.param_vars if k not in ("SVG Output Path", "Stacking Order")]
        def save_preset():
            name = self.preset_name_var.get().strip()
            if not name:
                self.status_var.set("Enter a preset name.")
                return
            preset_dict = {}
            for k in self.PRESET_PARAMS:
                v = self.param_vars[k]
                preset_dict[k] = v.get() if not isinstance(v, tk.BooleanVar) else bool(v.get())
            self.presets[name] = preset_dict
            with open(self.presets_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
            update_presets_menu()
            self.status_var.set(f"Preset '{name}' saved.")
        def load_preset(name):
            if name not in self.presets:
                self.status_var.set(f"Preset '{name}' not found.")
                return
            for k, v in self.presets[name].items():
                if k in self.param_vars:
                    var = self.param_vars[k]
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(v))
                    else:
                        var.set(v)
            self.status_var.set(f"Preset '{name}' loaded.")
        def update_preset():
            name = self.preset_menu_var.get()
            if not name or name not in self.presets:
                self.status_var.set("Select a preset to update.")
                return
            preset_dict = {}
            for k in self.PRESET_PARAMS:
                v = self.param_vars[k]
                preset_dict[k] = v.get() if not isinstance(v, tk.BooleanVar) else bool(v.get())
            self.presets[name] = preset_dict
            with open(self.presets_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
            update_presets_menu()
            self.status_var.set(f"Preset '{name}' updated.")
        def delete_preset():
            name = self.preset_menu_var.get()
            if not name or name not in self.presets:
                self.status_var.set("Select a preset to delete.")
                return
            del self.presets[name]
            with open(self.presets_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
            update_presets_menu()
            self.status_var.set(f"Preset '{name}' deleted.")
        def update_presets_menu():
            menu = self.preset_menu['menu']
            menu.delete(0, 'end')
            for name in self.presets:
                menu.add_command(label=name, command=lambda n=name: load_preset(n))
        # Preset GUI
        tk.Label(self.button_frame, text="Preset Name").grid(row=preview_row, column=0, sticky='e', pady=(10,2))
        tk.Entry(self.button_frame, textvariable=self.preset_name_var).grid(row=preview_row, column=1, pady=(10,2))
        tk.Button(self.button_frame, text="Save Preset", command=save_preset).grid(row=preview_row, column=2, pady=(10,2))
        preview_row += 1
        self.preset_menu = tk.OptionMenu(self.button_frame, self.preset_menu_var, *self.presets.keys(), command=load_preset)
        self.preset_menu.grid(row=preview_row, column=0, pady=2)
        tk.Button(self.button_frame, text="Update Preset", command=update_preset).grid(row=preview_row, column=1, pady=2)
        tk.Button(self.button_frame, text="Delete Preset", command=delete_preset).grid(row=preview_row, column=2, pady=2)
        preview_row += 1
        update_presets_menu()
        # --- History (Undo/Redo) ---
        self.history = []
        self.redo_stack = []
        def save_history():
            state = {k: v.get() if not isinstance(v, tk.BooleanVar) else bool(v.get()) for k, v in self.param_vars.items()}
            mod_state = {k: v.get() for k, v in self.modifier_vars.items()}
            self.history.append((state.copy(), mod_state.copy()))
            self.redo_stack.clear()
        def undo():
            if not self.history:
                self.status_var.set("No more undo steps.")
                return
            current_state = {k: v.get() if not isinstance(v, tk.BooleanVar) else bool(v.get()) for k, v in self.param_vars.items()}
            current_mod = {k: v.get() for k, v in self.modifier_vars.items()}
            self.redo_stack.append((current_state.copy(), current_mod.copy()))
            state, mod_state = self.history.pop()
            for k, v in state.items():
                var = self.param_vars[k]
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(v))
                else:
                    var.set(v)
            for k, v in mod_state.items():
                self.modifier_vars[k].set(v)
            self.status_var.set("Undo complete.")
        def redo():
            if not self.redo_stack:
                self.status_var.set("No more redo steps.")
                return
            current_state = {k: v.get() if not isinstance(v, tk.BooleanVar) else bool(v.get()) for k, v in self.param_vars.items()}
            current_mod = {k: v.get() for k, v in self.modifier_vars.items()}
            self.history.append((current_state.copy(), current_mod.copy()))
            state, mod_state = self.redo_stack.pop()
            for k, v in state.items():
                var = self.param_vars[k]
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(v))
                else:
                    var.set(v)
            for k, v in mod_state.items():
                self.modifier_vars[k].set(v)
            self.status_var.set("Redo complete.")
        # History GUI buttons
        tk.Button(self.button_frame, text="Undo", command=undo).grid(row=preview_row, column=0, sticky='ew')
        tk.Button(self.button_frame, text="Redo", command=redo).grid(row=preview_row, column=1, sticky='ew')
        preview_row += 1
        # Save history on parameter change
        for var in list(self.param_vars.values()) + list(self.modifier_vars.values()):
            var.trace_add('write', lambda *args: save_history())

        # Ensure subsequent button placement continues after preview/preset/history rows
        button_row = preview_row

        # Add droplet style toggle to button_frame
        self.droplet_style = tk.StringVar(value="painterly")
        # Keep label and option on the same row to avoid overlap
        tk.Label(self.button_frame, text="Droplet Style").grid(row=button_row, column=0, sticky='e')
        tk.OptionMenu(self.button_frame, self.droplet_style, "painterly", "organic").grid(row=button_row, column=1)
        button_row += 1

    def load_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if filename:
            img = Image.open(filename).convert('RGB')
            # Maintain aspect ratio, fit within max size (e.g. 360x240)
            max_w, max_h = 360, 240
            w, h = img.size
            scale = min(max_w / w, max_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            # Choose a resampling filter compatible with multiple Pillow versions
            try:
                resample_filter = Image.Resampling.LANCZOS  # Pillow >= 9
            except AttributeError:
                resample_filter = getattr(Image, "LANCZOS", Image.Resampling.BICUBIC)
            img = img.resize((new_w, new_h), resample=resample_filter)
            self.image = img
            self.img_np = img_as_float(np.array(img))
            self.show_image(self.img_np)

    def update_progress(self, percent, status=None):
        self.progress_var.set(percent)
        if status is not None:
            self.status_var.set(status)
        self.root.update_idletasks()

    def get_scale_factor(self, img_shape, crop_shape=(150, 150)):
        h, w = img_shape[:2]
        ch, cw = crop_shape
        if self.scaling_method.get() == "max":
            return max(h, w) / max(ch, cw)
        elif self.scaling_method.get() == "min":
            return min(h, w) / min(ch, cw)
        elif self.scaling_method.get() == "average":
            return ((h + w) / 2) / ((ch + cw) / 2)
        elif self.scaling_method.get() == "area":
            return (h * w) / (ch * cw)
        elif self.scaling_method.get() == "sqrt_area":
            return np.sqrt((h * w) / (ch * cw))
        else:
            return 1.0

    def run_vectoriser(self):
        """Launch the heavy vectorisation pipeline in a background thread.
        GUI updates (SVG preview, status/progress) are executed on the main thread
        in _run_vectoriser_finalize to avoid thread-safety issues.
        """
        if self.img_np is None:
            return
        # Disable the run button while working
        try:
            self.run_button.config(state=tk.DISABLED)
        except Exception:
            pass
        self.status_var.set("Starting background vectorisation...")
        self.update_progress(1)

        import threading

        def _worker():
            try:
                results = self.run_vectoriser_compute()
                # schedule finalization on the main thread
                self.root.after(0, lambda: self._run_vectoriser_finalize(results))
            except Exception as e:
                # Schedule an error update on the main thread
                self.root.after(0, lambda: self.status_var.set(f"Vectorisation failed: {e}"))
                self.root.after(0, lambda: self.update_progress(0))
            finally:
                # re-enable the run button on the main thread
                self.root.after(0, lambda: (self.run_button.config(state=tk.NORMAL) if hasattr(self, 'run_button') else None))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def run_vectoriser_compute(self):
        """Compute-only part of the vectorisation pipeline. This must not touch GUI state.
        Returns a dict with results needed to finalize (bridged_smoothed, all_droplets, colors, contour_indices, image shape).
        """
        # perform segmentation, quantization, contours, bridging, inflation, smoothing, droplet generation
        img_np = self.img_np
        # scale parameters
        crop_shape = (150, 150)
        scale_factor = self.get_scale_factor(img_np.shape, crop_shape=crop_shape)
        multiplier = self.param_vars["Segmentation Multiplier"].get()
        if multiplier == 0:
            multiplier = 1e-6
        noise_scale = max(1e-6, self.param_vars["Noise Scale"].get() * scale_factor / multiplier)
        blur_sigma = max(1e-6, self.param_vars["Blur Sigma"].get() * scale_factor / multiplier)
        compactness = max(1e-6, self.param_vars["Compactness"].get() * scale_factor / multiplier)

        labels = noise_watershed(img_np, scale=noise_scale, blur_sigma=blur_sigma, compactness=compactness)

        # Color quantization (modifier)
        if self.modifier_vars["Color Quantization"].get():
            quantized_image, _ = quantize_image_colors((img_np * 255).astype(np.uint8), self.param_vars["Max Colors"].get())
        else:
            quantized_image = (img_np * 255).astype(np.uint8)

        # Extract contours and colors
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

        # Bridging
        if self.modifier_vars["Bridging"].get():
            bridged = improved_bridge_contours(
                contours, centroids, colors,
                bridge_distance=self.param_vars["Bridge Distance"].get(),
                color_tolerance=self.param_vars["Color Tolerance"].get(),
                proximity_threshold=self.param_vars["Proximity Threshold"].get(),
                falloff_radius=self.param_vars["Falloff Radius"].get(),
                max_curvature=np.radians(self.param_vars["Max Curvature"].get())
            )
        else:
            bridged = contours

        # Inflation
        N_pre = len(bridged)
        inflation_stacking = self.param_vars["Inflation Proportional to Stacking"].get()
        inflation_amount = self.param_vars["Blob Inflation Amount"].get()
        far_point_factor = self.param_vars["Far Point Inflation Factor"].get()
        inflation_scaled_contours = []
        for order_idx, contour in enumerate(bridged):
            if self.modifier_vars["Inflation"].get():
                if inflation_stacking:
                    if N_pre > 1:
                        stack_scale = np.log1p(order_idx) / np.log1p(N_pre - 1)
                    else:
                        stack_scale = 1.0
                    scaled_inflation = inflation_amount * stack_scale
                    scaled_far_point_factor = 1.0 + (far_point_factor - 1.0) * stack_scale
                else:
                    scaled_inflation = inflation_amount
                    scaled_far_point_factor = far_point_factor
                inflated = inflate_contour(contour, scaled_inflation, scaled_far_point_factor)
            else:
                inflated = contour
            inflation_scaled_contours.append(inflated)

        # Smoothing
        if self.modifier_vars["Smoothing"].get():
            bridged_smoothed = [
                laplacian_smooth(c, iterations=self.param_vars["Smooth Iterations"].get(), alpha=self.param_vars["Smooth Alpha"].get())
                for c in inflation_scaled_contours
            ]
        else:
            bridged_smoothed = inflation_scaled_contours

        # Droplet generation
        all_droplets = []
        droplet_density = self.param_vars["Droplet Density"].get()
        droplet_min_distance = self.param_vars["Droplet Min Distance"].get()
        droplet_max_distance = self.param_vars["Droplet Max Distance"].get()
        droplet_size_mean = self.param_vars["Droplet Size Mean"].get()
        droplet_size_std = self.param_vars["Droplet Size Std"].get()
        droplet_spread = np.radians(self.param_vars["Droplet Spread"].get())

        droplet_organic_min_brightness = self.param_vars["Droplet Organic Min Brightness"].get()
        droplet_organic_density = self.param_vars["Droplet Organic Density"].get()
        droplet_organic_strength = self.param_vars["Droplet Organic Strength"].get()
        droplet_organic_jitter = self.param_vars["Droplet Organic Jitter"].get()
        droplet_organic_elongation = self.param_vars["Droplet Organic Elongation"].get()
        droplet_organic_percent_per_blob = self.param_vars.get("Droplet Organic Percent Per Blob", tk.DoubleVar(value=100.0)).get()

        centroids_smoothed = [np.mean(c, axis=0) if len(c) > 0 else np.array([0.0, 0.0]) for c in bridged_smoothed]
        n_blobs = len(bridged_smoothed)

        def luminance_rgb(rgb):
            r, g, b = rgb
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        for i, contour in enumerate(bridged_smoothed):
            if contour is None or len(contour) == 0:
                continue
            color_val = colors[i] if i < len(colors) else (0, 0, 0)
            current_centroid = centroids_smoothed[i]

            if self.droplet_style.get() == "organic":
                if luminance_rgb(color_val) < droplet_organic_min_brightness:
                    continue
                neighbor_indices = [j for j in range(n_blobs) if j != i]
                n_neighbors = len(neighbor_indices)
                if n_neighbors == 0:
                    continue
                n_selected = max(1, int(np.ceil((droplet_organic_percent_per_blob / 100.0) * n_neighbors)))
                n_selected = min(n_selected, n_neighbors)
                selected_neighbors = list(np.random.choice(neighbor_indices, n_selected, replace=False))
                for j in selected_neighbors:
                    neighbor_centroid = centroids_smoothed[j]
                    direction = neighbor_centroid - current_centroid
                    norm = np.linalg.norm(direction)
                    if norm == 0:
                        continue
                    direction = direction / norm
                    droplets = generate_organic_droplets(
                        contour=contour,
                        direction=direction,
                        num_droplets=droplet_organic_density,
                        min_dist=droplet_min_distance,
                        max_dist=droplet_max_distance,
                        size_mean=droplet_size_mean,
                        size_std=droplet_size_std,
                        spread_angle=droplet_spread,
                        organic_strength=droplet_organic_strength,
                        jitter_amount=droplet_organic_jitter,
                        elongation=droplet_organic_elongation,
                        simplify_tol=self.param_vars["Simplify Tolerance"].get()
                    )
                    for poly in droplets:
                        all_droplets.append((poly, color_val))
            else:
                mean_point = np.mean(contour, axis=0)
                direction = mean_point - current_centroid
                norm = np.linalg.norm(direction)
                if norm == 0:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = direction / norm
                if self.param_vars["Painterly Use SVG Ellipses"].get():
                    prim = self.param_vars.get("Painterly SVG Primitive", tk.StringVar(value="ellipse")).get()
                    if prim == "ellipse":
                        droplets = generate_painterly_ellipses(
                            contour=contour,
                            direction=direction,
                            num_droplets=droplet_density,
                            min_dist=droplet_min_distance,
                            max_dist=droplet_max_distance,
                            size_mean=droplet_size_mean,
                            size_std=droplet_size_std,
                            spread_angle=droplet_spread,
                            rotation_offset=self.param_vars["Droplet Global Rotation"].get()
                        )
                        for ell in droplets:
                            all_droplets.append((ell, color_val))
                    elif prim == "rect":
                        droplets = generate_painterly_rects(
                            contour=contour,
                            direction=direction,
                            num_droplets=droplet_density,
                            min_dist=droplet_min_distance,
                            max_dist=droplet_max_distance,
                            size_mean=droplet_size_mean,
                            size_std=droplet_size_std,
                            spread_angle=droplet_spread,
                            rotation_offset=self.param_vars["Droplet Global Rotation"].get(),
                            horizontal=bool(self.param_vars.get("Painterly Rect Horizontal", tk.BooleanVar(value=False)).get())
                        )
                        for r in droplets:
                            all_droplets.append((r, color_val))
                else:
                    droplets = generate_droplets(
                        contour=contour,
                        direction=direction,
                        num_droplets=droplet_density,
                        min_dist=droplet_min_distance,
                        max_dist=droplet_max_distance,
                        size_mean=droplet_size_mean,
                        size_std=droplet_size_std,
                        spread_angle=droplet_spread,
                        simplify_tol=self.param_vars["Simplify Tolerance"].get()
                    )
                    for ellipse in droplets:
                        all_droplets.append((ellipse, color_val))

        # stacking order
        def contour_area(contour):
            x = contour[:, 1]
            y = contour[:, 0]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        def luminance(rgb):
            r, g, b = rgb
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        _shape = tuple(getattr(self.img_np, 'shape', (0, 0)))
        if len(_shape) >= 2:
            h, w = int(_shape[0]), int(_shape[1])
        elif len(_shape) == 1:
            h, w = int(_shape[0]), 1
        else:
            h, w = 0, 0
        contour_indices = list(range(len(bridged_smoothed)))
        shape_order = self.param_vars["Stacking Order"].get()
        if shape_order == "area":
            areas = [contour_area(np.array(c)) for c in bridged_smoothed]
            contour_indices = np.argsort(-np.array(areas))
        elif shape_order == "area_reverse":
            areas = [contour_area(np.array(c)) for c in bridged_smoothed]
            contour_indices = np.argsort(np.array(areas))
        elif shape_order == "brightness":
            lums = [luminance(c) for c in colors]
            contour_indices = np.argsort(lums)
        elif shape_order == "brightness_reverse":
            lums = [luminance(c) for c in colors]
            contour_indices = np.argsort(-np.array(lums))
        elif shape_order == "position_y":
            ys = [np.mean(np.array(c)[:, 0]) for c in bridged_smoothed]
            contour_indices = np.argsort(ys)
        elif shape_order == "position_y_reverse":
            ys = [np.mean(np.array(c)[:, 0]) for c in bridged_smoothed]
            contour_indices = np.argsort(-np.array(ys))
        elif shape_order == "position_x":
            xs = [np.mean(np.array(c)[:, 1]) for c in bridged_smoothed]
            contour_indices = np.argsort(xs)
        elif shape_order == "position_x_reverse":
            xs = [np.mean(np.array(c)[:, 1]) for c in bridged_smoothed]
            contour_indices = np.argsort(-np.array(xs))
        elif shape_order == "position_centre":
            center = np.array([h / 2, w / 2])
            dists = [np.linalg.norm(np.mean(np.array(c), axis=0) - center) for c in bridged_smoothed]
            contour_indices = np.argsort(-np.array(dists))
        elif shape_order == "position_centre_reverse":
            center = np.array([h / 2, w / 2])
            dists = [np.linalg.norm(np.mean(np.array(c), axis=0) - center) for c in bridged_smoothed]
            contour_indices = np.argsort(np.array(dists))

        return {
            'bridged_smoothed': bridged_smoothed,
            'all_droplets': all_droplets,
            'colors': colors,
            'contour_indices': contour_indices,
            'h': h,
            'w': w
        }

    def _run_vectoriser_finalize(self, results):
        """Run on the main thread: write SVG, optionally draw vector preview, and render final preview image.
        """
        bridged_smoothed = results.get('bridged_smoothed', [])
        all_droplets = results.get('all_droplets', [])
        colors = results.get('colors', [])
        contour_indices = list(results.get('contour_indices', list(range(len(bridged_smoothed)))))
        h = int(results.get('h', 0))
        w = int(results.get('w', 0))

        # SVG generation (guarded imports)
        try:
            import svgwrite
        except Exception:
            svgwrite = None

        if svgwrite is not None:
            try:
                dwg = svgwrite.Drawing(filename=self.svg_path.get(), size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
                dwg.defs.add(dwg.style("path { stroke: none; stroke-width: 0; }"))
                for idx in contour_indices:
                    contour = bridged_smoothed[idx]
                    color_val = colors[idx] if idx < len(colors) else (0, 0, 0)
                    hex_color = svgwrite.rgb(r=int(color_val[0]), g=int(color_val[1]), b=int(color_val[2]))
                    path_data = contour_to_svg_path(contour, simplify_tol=self.param_vars["Simplify Tolerance"].get())
                    if path_data:
                        dwg.add(dwg.path(d=path_data, fill=hex_color))
                for droplet_obj, color_val in all_droplets:
                    hex_color = svgwrite.rgb(r=int(color_val[0]), g=int(color_val[1]), b=int(color_val[2]))
                    if isinstance(droplet_obj, dict):
                        if droplet_obj.get('type') == 'ellipse':
                            cx = float(droplet_obj.get('cx', 0.0))
                            cy = float(droplet_obj.get('cy', 0.0))
                            rx = float(droplet_obj.get('rx', 1.0))
                            ry = float(droplet_obj.get('ry', 1.0))
                            angle = float(droplet_obj.get('angle', 0.0))
                            el = dwg.ellipse(center=(cx, cy), r=(rx, ry), fill=hex_color)
                            if angle != 0.0:
                                el.rotate(angle, center=(cx, cy))
                            dwg.add(el)
                        elif droplet_obj.get('type') == 'rect':
                            cx = float(droplet_obj.get('cx', 0.0))
                            cy = float(droplet_obj.get('cy', 0.0))
                            w_rect = float(droplet_obj.get('w', 1.0))
                            h_rect = float(droplet_obj.get('h', 1.0))
                            angle = float(droplet_obj.get('angle', 0.0))
                            insert_x = cx - (w_rect / 2.0)
                            insert_y = cy - (h_rect / 2.0)
                            el = dwg.rect(insert=(insert_x, insert_y), size=(w_rect, h_rect), fill=hex_color)
                            if angle != 0.0:
                                el.rotate(angle, center=(cx, cy))
                            dwg.add(el)
                    else:
                        try:
                            path_d = contour_to_svg_path(droplet_obj, simplify_tol=self.param_vars["Simplify Tolerance"].get())
                            if path_d:
                                dwg.add(dwg.path(d=path_d, fill=hex_color))
                        except Exception:
                            pass
                try:
                    dwg.save()
                except Exception:
                    try:
                        with open(self.svg_path.get(), 'w', encoding='utf-8') as _f:
                            _f.write(dwg.tostring())
                    except Exception:
                        pass
            except Exception:
                # fallback to raw SVG writing if svgwrite usage fails
                svgwrite = None

        if svgwrite is None:
            try:
                with open(self.svg_path.get(), 'w', encoding='utf-8') as _f:
                    _f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                    _f.write(f'<svg width="{w}px" height="{h}px" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">\n')
                    for idx in contour_indices:
                        contour = bridged_smoothed[idx]
                        color_val = colors[idx] if idx < len(colors) else (0, 0, 0)
                        path_data = contour_to_svg_path(contour, simplify_tol=self.param_vars["Simplify Tolerance"].get())
                        if path_data:
                            _f.write(f'<path d="{path_data}" fill="rgb({int(color_val[0])},{int(color_val[1])},{int(color_val[2])})"/>\n')
                    for droplet_obj, color_val in all_droplets:
                        if isinstance(droplet_obj, dict) and droplet_obj.get('type') == 'ellipse':
                            cx = float(droplet_obj.get('cx', 0.0))
                            cy = float(droplet_obj.get('cy', 0.0))
                            rx = float(droplet_obj.get('rx', 1.0))
                            ry = float(droplet_obj.get('ry', 1.0))
                            angle = float(droplet_obj.get('angle', 0.0))
                            _f.write(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="rgb({int(color_val[0])},{int(color_val[1])},{int(color_val[2])})" transform="rotate({angle} {cx} {cy})"/>\n')
                        elif isinstance(droplet_obj, dict) and droplet_obj.get('type') == 'rect':
                            cx = float(droplet_obj.get('cx', 0.0))
                            cy = float(droplet_obj.get('cy', 0.0))
                            w_rect = float(droplet_obj.get('w', 1.0))
                            h_rect = float(droplet_obj.get('h', 1.0))
                            angle = float(droplet_obj.get('angle', 0.0))
                            insert_x = cx - (w_rect / 2.0)
                            insert_y = cy - (h_rect / 2.0)
                            _f.write(f'<rect x="{insert_x}" y="{insert_y}" width="{w_rect}" height="{h_rect}" fill="rgb({int(color_val[0])},{int(color_val[1])},{int(color_val[2])})" transform="rotate({angle} {cx} {cy})"/>\n')
                        else:
                            path_d = contour_to_svg_path(droplet_obj, simplify_tol=self.param_vars["Simplify Tolerance"].get())
                            if path_d:
                                _f.write(f'<path d="{path_d}" fill="rgb({int(color_val[0])},{int(color_val[1])},{int(color_val[2])})"/>\n')
                    _f.write('</svg>\n')
            except Exception:
                pass

        # Optionally draw a native vector representation directly into the main matplotlib axes
        try:
            if self.param_vars.get("Enable Vector Preview", tk.BooleanVar(value=False)).get():
                try:
                    self.draw_vectors_to_axes(bridged_smoothed, all_droplets, colors, contour_indices=contour_indices, img_shape=(h, w), use_native_ellipses=self.param_vars["Painterly Use SVG Ellipses"].get(), simplify_tol=self.param_vars["Simplify Tolerance"].get())
                except Exception:
                    pass
        except Exception:
            pass

        self.update_progress(100, "SVG file saved. Vectorisation complete!")

        # Try to render the full SVG into the main image view (right column) using cairosvg if available
        try:
            try:
                import cairosvg
            except Exception:
                cairosvg = None
            if cairosvg is not None:
                try:
                    png_bytes = cairosvg.svg2png(url=self.svg_path.get(), output_width=w, output_height=h)
                    if not png_bytes:
                        if self.img_np is not None:
                            self.show_image(self.img_np)
                    else:
                        if isinstance(png_bytes, memoryview) or isinstance(png_bytes, bytearray):
                            png_bytes = bytes(png_bytes)
                        im = Image.open(BytesIO(png_bytes)).convert('RGBA')
                        im_np = img_as_float(np.array(im.convert('RGB')))
                        self.show_image(im_np)
                except Exception:
                    if self.img_np is not None:
                        self.show_image(self.img_np)
            else:
                if self.img_np is not None:
                    self.show_image(self.img_np)
        except Exception as e:
            self.status_var.set(f"SVG saved, preview render failed: {str(e)}")

        self.update_progress(0, "Ready.")

    def run_preview(self):
        """Compute and render a lightweight 150x150 centre-crop preview applying the selected
        segmentation -> bridging -> inflation -> smoothing modifiers (no droplets). This is
        intentionally lightweight so the UI remains responsive and the app can be closed.
        """
        try:
            if self.img_np is None:
                self.status_var.set("No image loaded for preview.")
                return
            self.update_progress(5, "Starting preview...")
            ch, cw = 150, 150
            # Robustly obtain image shape so type-checkers don't complain if shape has unexpected length
            img_shape = getattr(self.img_np, 'shape', ())
            if img_shape is None:
                img_shape = ()
            if len(img_shape) >= 2:
                h, w = int(img_shape[0]), int(img_shape[1])
            elif len(img_shape) == 1:
                h, w = int(img_shape[0]), 1
            else:
                h, w = 0, 0
            y0 = max(0, (h // 2) - (ch // 2))
            x0 = max(0, (w // 2) - (cw // 2))
            y1 = min(h, y0 + ch)
            x1 = min(w, x0 + cw)
            crop = self.img_np[y0:y1, x0:x1]
            if crop.size == 0:
                self.status_var.set("Preview crop is empty.")
                return
            # scale parameters for the crop
            scale_factor = self.get_scale_factor(self.img_np.shape, crop_shape=(ch, cw))
            multiplier = self.param_vars["Segmentation Multiplier"].get()
            if multiplier == 0:
                multiplier = 1e-6
            noise_scale = max(1e-6, self.param_vars["Noise Scale"].get() * scale_factor / multiplier)
            blur_sigma = max(1e-6, self.param_vars["Blur Sigma"].get() * scale_factor / multiplier)
            compactness = max(1e-6, self.param_vars["Compactness"].get() * scale_factor / multiplier)

            self.update_progress(20, "Running segmentation (preview)...")
            labels = noise_watershed(crop, scale=noise_scale, blur_sigma=blur_sigma, compactness=compactness)

            self.update_progress(40, "Quantizing (preview)...")
            if self.modifier_vars.get("Color Quantization", tk.BooleanVar(value=True)).get():
                quantized_image, _ = quantize_image_colors((crop * 255).astype(np.uint8), self.param_vars["Max Colors"].get())
            else:
                quantized_image = (crop * 255).astype(np.uint8)

            self.update_progress(60, "Extracting contours (preview)...")
            unique_labels = np.unique(labels)
            contours = []
            colors = []
            centroids = []
            for seg_id in unique_labels:
                mask = (labels == seg_id)
                colors_in_mask = quantized_image[mask]
                if colors_in_mask.size == 0:
                    continue
                unique_colors, counts = np.unique(colors_in_mask.reshape(-1, 3), axis=0, return_counts=True)
                main_color = tuple(unique_colors[np.argmax(counts)])
                found = find_contours(mask.astype(float), level=0.5)
                if found:
                    contour = found[0]
                    contours.append(contour)
                    centroids.append(np.mean(contour, axis=0))
                    colors.append(main_color)

            self.update_progress(70, "Applying bridging/inflation/smoothing (preview)...")
            if self.modifier_vars.get("Bridging", tk.BooleanVar(value=True)).get():
                bridged = improved_bridge_contours(
                    contours, centroids, colors,
                    bridge_distance=self.param_vars["Bridge Distance"].get(),
                    color_tolerance=self.param_vars["Color Tolerance"].get(),
                    proximity_threshold=self.param_vars["Proximity Threshold"].get(),
                    falloff_radius=self.param_vars["Falloff Radius"].get(),
                    max_curvature=np.radians(self.param_vars["Max Curvature"].get())
                )
            else:
                bridged = contours

            # Inflation
            inflation_scaled_contours = []
            inflation_amount = self.param_vars["Blob Inflation Amount"].get()
            far_point_factor = self.param_vars["Far Point Inflation Factor"].get()
            inflation_stacking = self.param_vars["Inflation Proportional to Stacking"].get()
            N_pre = len(bridged)
            for order_idx, contour in enumerate(bridged):
                if self.modifier_vars.get("Inflation", tk.BooleanVar(value=True)).get():
                    if inflation_stacking:
                        if N_pre > 1:
                            stack_scale = np.log1p(order_idx) / np.log1p(max(1, N_pre-1))
                        else:
                            stack_scale = 1.0
                        scaled_inflation = inflation_amount * stack_scale
                        scaled_far_point_factor = 1.0 + (far_point_factor - 1.0) * stack_scale
                    else:
                        scaled_inflation = inflation_amount
                        scaled_far_point_factor = far_point_factor
                    try:
                        inflated = inflate_contour(contour, scaled_inflation, scaled_far_point_factor)
                    except Exception:
                        inflated = contour
                else:
                    inflated = contour
                inflation_scaled_contours.append(inflated)

            # Smoothing
            if self.modifier_vars.get("Smoothing", tk.BooleanVar(value=True)).get():
                bridged_smoothed = [
                    laplacian_smooth(c, iterations=self.param_vars["Smooth Iterations"].get(), alpha=self.param_vars["Smooth Alpha"].get())
                    for c in inflation_scaled_contours
                ]
            else:
                bridged_smoothed = inflation_scaled_contours

            # Draw raster preview: crop image + filled contours
            try:
                self.ax.clear()
                # show crop (convert to displayable range)
                disp = np.clip(crop, 0.0, 1.0)
                self.ax.imshow(disp)
                # overlay filled contours
                for idx, contour in enumerate(bridged_smoothed):
                    if contour is None or len(contour) < 3:
                        continue
                    col = colors[idx] if idx < len(colors) else (0, 0, 0)
                    # convert 0-255 to 0-1
                    face = (col[0]/255.0, col[1]/255.0, col[2]/255.0, 1.0)
                    try:
                        verts = [(pt[1], pt[0]) for pt in contour]
                        poly = mpatches.Polygon(verts, closed=True, facecolor=face, edgecolor=None)
                        self.ax.add_patch(poly)
                    except Exception:
                        # fallback: plot contour line
                        xs = [pt[1] for pt in contour]
                        ys = [pt[0] for pt in contour]
                        self.ax.plot(xs, ys, color=face[:3], linewidth=1)
                        self.ax.plot(xs, ys, color=face[:3], linewidth=1)
                self.ax.set_title('Preview (150x150)')
                self.ax.axis('off')
                self.fig.tight_layout()
                self.canvas_plot.draw()
            except Exception as e:
                self.status_var.set(f"Preview render failed: {str(e)}")

            self.update_progress(100, "Preview complete.")
        except Exception as e:
            self.status_var.set(f"Preview error: {str(e)}")
        finally:
            # allow UI to refresh
            self.root.update_idletasks()
            self.update_progress(0, "Ready.")

    def show_image(self, img_np):
        self.ax.clear()
        try:
            if img_np is None:
                return
            if img_np.ndim == 2:
                self.ax.imshow(np.clip(img_np, 0.0, 1.0), cmap='gray')
            else:
                # assume RGB float [0,1] or uint8 [0,255]
                arr = np.asarray(img_np)
                if arr.dtype == np.uint8:
                    arr = arr.astype(np.float32) / 255.0
                self.ax.imshow(np.clip(arr, 0.0, 1.0))
        except Exception:
            # final fallback: try converting via PIL
            try:
                im = Image.fromarray((np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8))
                self.ax.imshow(im)
            except Exception:
                pass
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas_plot.draw()

    def draw_vectors_to_axes(self, bridged_smoothed, all_droplets, colors, contour_indices=None,
                             img_shape=None, use_native_ellipses=False, simplify_tol=0.5):
        """Draw contours and droplets as vector primitives directly into the main matplotlib axes.

        - bridged_smoothed: list of contours (arrays of (row, col)).
        - all_droplets: list of tuples (droplet_obj, color_val) where droplet_obj is either
          a polygon (array of (row, col)) or a dict describing an ellipse when use_native_ellipses is True.
        - colors: list of RGB tuples (0-255).
        - contour_indices: optional ordering of contours to draw.
        - img_shape: (h, w) of the canvas/image coordinates.
        """
        try:
            self.ax.clear()
            h = img_shape[0] if img_shape is not None and len(img_shape) > 0 else (getattr(self.img_np, 'shape', (0,0))[0] if self.img_np is not None else 0)
            w = img_shape[1] if img_shape is not None and len(img_shape) > 1 else (getattr(self.img_np, 'shape', (0,0))[1] if self.img_np is not None else 0)
            # show background image if available
            if self.img_np is not None:
                try:
                    bg = self.img_np
                    if bg.dtype == np.uint8:
                        bg = bg.astype(np.float32) / 255.0
                    self.ax.imshow(np.clip(bg, 0.0, 1.0))
                except Exception:
                    pass
            # draw contours in order
            indices = contour_indices if contour_indices is not None else list(range(len(bridged_smoothed)))
            for idx in indices:
                try:
                    contour = bridged_smoothed[idx]
                    if contour is None or len(contour) < 3:
                        continue
                    col = colors[idx] if idx < len(colors) else (0, 0, 0)
                    face = (col[0]/255.0, col[1]/255.0, col[2]/255.0, 1.0)
                    verts = [(pt[1], pt[0]) for pt in contour]
                    poly = mpatches.Polygon(verts, closed=True, facecolor=face, edgecolor=None)
                    self.ax.add_patch(poly)
                except Exception:
                    continue
            for droplet_obj, color_val in all_droplets:
                try:
                    face = (color_val[0]/255.0, color_val[1]/255.0, color_val[2]/255.0, 1.0)
                    if isinstance(droplet_obj, dict) and droplet_obj.get('type') == 'ellipse':
                        cx = float(droplet_obj.get('cx', 0.0))
                        cy = float(droplet_obj.get('cy', 0.0))
                        rx = float(droplet_obj.get('rx', 1.0))
                        ry = float(droplet_obj.get('ry', 1.0))
                        angle = float(droplet_obj.get('angle', 0.0))
                        el = mpatches.Ellipse((cx, cy), width=rx*2, height=ry*2, angle=angle, facecolor=face, edgecolor=None)
                        self.ax.add_patch(el)
                    elif isinstance(droplet_obj, dict) and droplet_obj.get('type') == 'rect':
                        cx = float(droplet_obj.get('cx', 0.0))
                        cy = float(droplet_obj.get('cy', 0.0))
                        w_rect = float(droplet_obj.get('w', 1.0))
                        h_rect = float(droplet_obj.get('h', 1.0))
                        angle = float(droplet_obj.get('angle', 0.0))
                        insert_x = cx - (w_rect / 2.0)
                        insert_y = cy - (h_rect / 2.0)
                        # Create rectangle without angle then rotate around its center using Affine2D so rotation
                        # matches SVG rotation around center
                        rect = mpatches.Rectangle((insert_x, insert_y), width=w_rect, height=h_rect, facecolor=face, edgecolor=None)
                        try:
                            from matplotlib import transforms
                            rot = transforms.Affine2D().rotate_deg_around(cx, cy, angle)
                            rect.set_transform(rot + self.ax.transData)
                        except Exception:
                            # Fallback: set angle param (may rotate about lower-left in some backends)
                            rect.set_angle(angle)
                        self.ax.add_patch(rect)
                    else:
                        pts = np.asarray(droplet_obj)
                        if pts.size == 0:
                            continue
                        verts = [(pt[1], pt[0]) for pt in pts]
                        poly = mpatches.Polygon(verts, closed=True, facecolor=face, edgecolor=None)
                        self.ax.add_patch(poly)
                except Exception:
                    continue
                    continue
            self.ax.axis('off')
            self.fig.tight_layout()
            self.canvas_plot.draw()
        except Exception:
            # make sure any drawing failure does not block the app
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = OrganicSegmentationGUI(root)
    root.mainloop()