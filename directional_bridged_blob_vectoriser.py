import numpy as np
import svgwrite
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.measure import find_contours, perimeter
from scipy.spatial import KDTree
from scipy.ndimage import label as nd_label, binary_dilation
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
import argparse

def load_image(image_path, resize_width=None):
    image = Image.open(image_path).convert('RGB')
    if resize_width:
        w_percent = resize_width / float(image.width)
        new_height = int(float(image.height) * w_percent)
        image = image.resize((resize_width, new_height), Image.LANCZOS)
    return np.array(image)

def get_color_mask(segments, segment_colors, target_color):
    mask = np.zeros_like(segments, dtype=bool)
    for seg_id, color in segment_colors.items():
        if color == target_color:
            mask[segments == seg_id] = True
    return mask

def selective_merge_slic_blobs(segments, quantized_image, area_threshold=120):
    slic_ids = np.unique(segments)
    segment_colors = {}
    for sid in slic_ids:
        mask = (segments == sid)
        colors_in_mask = quantized_image[mask]
        unique_colors, counts = np.unique(colors_in_mask.reshape(-1, 3), axis=0, return_counts=True)
        main_color = tuple(unique_colors[np.argmax(counts)])
        segment_colors[sid] = main_color

    merged_labels = np.copy(segments)
    label_img = merged_labels
    unique_colors = list({v for v in segment_colors.values()})
    for color in unique_colors:
        mask = get_color_mask(label_img, segment_colors, color)
        cc_labels, num_cc = nd_label(mask)
        cc_areas = [(cc_labels == i).sum() for i in range(1, num_cc+1)]
        for i in range(1, num_cc+1):
            area = cc_areas[i-1]
            if area >= area_threshold:
                continue
            region = (cc_labels == i)
            dilated = binary_dilation(region)
            border = dilated & (~region)
            neighbor_colors = quantized_image[border]
            if np.any(np.all(neighbor_colors == color, axis=1)):
                neighbor_labels = label_img[border]
                candidate_labels = [lbl for lbl in np.unique(neighbor_labels)
                                    if segment_colors.get(lbl, None) == color]
                if candidate_labels:
                    neighbor_areas = [np.sum(label_img == lbl) for lbl in candidate_labels]
                    target_lbl = candidate_labels[np.argmax(neighbor_areas)]
                    label_img[region] = target_lbl
    return label_img, segment_colors

def quantize_image_colors(image, max_colors):
    if isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=max_colors, n_init=10).fit(pixels)
    new_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
    return new_pixels.reshape(image.shape), kmeans.cluster_centers_

def contour_to_svg_path(contour, scale=1.0):
    path_str = "M " + " L ".join(f"{x*scale:.2f},{y*scale:.2f}" for y, x in contour)
    return path_str + " Z"

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

def create_directional_bridged_svg(
    image, output_path="directional_bridged_blobs.svg",
    n_segments=40, compactness=10, max_colors=8,
    bridge_distance=5.0, color_tolerance=10.0,
    proximity_threshold=50.0, falloff_radius=5,
    max_curvature=np.radians(160),
    smooth_iterations=3, smooth_alpha=0.3,
    progress_callback=None,
    shape_order="area",
    enable_grid_merge=True,
    area_threshold=1000,
    compactness_threshold=1.35,
    inflation_amount=0.0,
    far_point_factor=1.0,
    inflation_stacking=True  # <-- new argument
):
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
        img_np = np.array(img)
    else:
        img_np = image
    h, w, _ = img_np.shape

    def report(val):
        if progress_callback:
            progress_callback(val)
    report(0)
    quantized_image, _ = quantize_image_colors(img_np, max_colors)
    report(10)

    image_float = img_as_float(quantized_image)
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=1)
    merged_labels, segment_colors = selective_merge_slic_blobs(segments, quantized_image, area_threshold=120)
    report(20)

    unique_segments = np.unique(merged_labels)
    contours, centroids, colors = [], [], []
    n_segs = len(unique_segments)

    def compactness(mask):
        area = np.sum(mask)
        perim = perimeter(mask)
        if area == 0:
            return 9999
        return (perim ** 2) / (4 * np.pi * area)

    if enable_grid_merge:
        area_threshold = 1000
        compactness_threshold = 1.35
        merged_labels = np.copy(merged_labels)
        num_merges = 1

        while num_merges > 0:
            num_merges = 0
            unique_segments = np.unique(merged_labels)
            segment_colors = {}
            segment_areas = {}
            segment_compactness = {}

            for seg_id in unique_segments:
                mask = (merged_labels == seg_id)
                colors_in_mask = quantized_image[mask]
                unique_colors, counts = np.unique(colors_in_mask.reshape(-1, 3), axis=0, return_counts=True)
                main_color = tuple(unique_colors[np.argmax(counts)])
                segment_colors[seg_id] = main_color
                segment_areas[seg_id] = np.sum(mask)
                segment_compactness[seg_id] = compactness(mask)

            for seg_id in unique_segments:
                area = segment_areas[seg_id]
                comp = segment_compactness[seg_id]
                if not (area < area_threshold and comp < compactness_threshold):
                    continue
                mask = (merged_labels == seg_id)
                border = binary_dilation(mask) & (~mask)
                neighbor_sids = np.unique(merged_labels[border])
                for nsid in neighbor_sids:
                    if nsid != seg_id and segment_colors.get(nsid) == segment_colors[seg_id]:
                        merged_labels[merged_labels == seg_id] = nsid
                        num_merges += 1
                        break
    else:
        merged_labels = np.copy(segments)
        unique_segments = np.unique(merged_labels)
        segment_colors = {}
        for seg_id in unique_segments:
            mask = (merged_labels == seg_id)
            colors_in_mask = quantized_image[mask]
            unique_colors, counts = np.unique(colors_in_mask.reshape(-1, 3), axis=0, return_counts=True)
            main_color = tuple(unique_colors[np.argmax(counts)])
            segment_colors[seg_id] = main_color

    unique_merged_segments = np.unique(merged_labels)
    contours, centroids, colors = [], [], []
    n_segs = len(unique_merged_segments)

    for idx, seg_id in enumerate(unique_merged_segments):
        mask = merged_labels == seg_id
        main_color = segment_colors.get(seg_id, (0,0,0))
        found_contours = find_contours(mask.astype(float), level=0.5)
        if found_contours:
            contour = found_contours[0]
            contours.append(contour)
            centroids.append(np.mean(contour, axis=0))
            colors.append(main_color)
        report(20 + int(30 * (idx+1) / n_segs))

    bridged = improved_bridge_contours(
        contours, centroids, colors,
        bridge_distance, color_tolerance,
        proximity_threshold, falloff_radius,
        max_curvature
    )
    report(60)

    N = len(bridged)
    bridged_smoothed = []
    for idx, c in enumerate(bridged):
        bridged_smoothed.append(
            laplacian_smooth(c, iterations=smooth_iterations, alpha=smooth_alpha)
        )
        report(60 + int(25 * (idx+1) / N))

    # Prepare SVG drawing
    dwg = svgwrite.Drawing(filename=output_path, size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
    dwg.defs.add(dwg.style("path { stroke: none; stroke-width: 0; }"))

    # Calculate stacking order indices before inflation scaling
    def contour_area(contour):
        x = contour[:, 1]
        y = contour[:, 0]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    contour_indices = list(range(len(bridged_smoothed)))
    def luminance(rgb):
        r, g, b = rgb
        return 0.2126*r + 0.7152*g + 0.0722*b
    h, w, _ = img_np.shape
    contour_indices = list(range(len(bridged_smoothed)))
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
        ys = [np.mean(np.array(c)[:,0]) for c in bridged_smoothed]
        contour_indices = np.argsort(ys)
    elif shape_order == "position_y_reverse":
        ys = [np.mean(np.array(c)[:,0]) for c in bridged_smoothed]
        contour_indices = np.argsort(-np.array(ys))
    elif shape_order == "position_x":
        xs = [np.mean(np.array(c)[:,1]) for c in bridged_smoothed]
        contour_indices = np.argsort(xs)
    elif shape_order == "position_x_reverse":
        xs = [np.mean(np.array(c)[:,1]) for c in bridged_smoothed]
        contour_indices = np.argsort(-np.array(xs))
    elif shape_order == "position_centre":
        center = np.array([h/2, w/2])
        dists = [np.linalg.norm(np.mean(np.array(c), axis=0) - center) for c in bridged_smoothed]
        contour_indices = np.argsort(-np.array(dists))
    elif shape_order == "position_centre_reverse":
        center = np.array([h/2, w/2])
        dists = [np.linalg.norm(np.mean(np.array(c), axis=0) - center) for c in bridged_smoothed]
        contour_indices = np.argsort(np.array(dists))

    # Calculate inflation scaling for stacking order
    N = len(bridged_smoothed)
    inflation_scaled_contours = []
    for order_idx, idx in enumerate(contour_indices):
        # Logarithmic scaling: bottommost gets zero, topmost gets full effect
        if N > 1:
            stack_scale = np.log1p(order_idx) / np.log1p(N-1)  # 0 for bottom, 1 for top
        else:
            stack_scale = 1.0
        scaled_inflation = inflation_amount * stack_scale
        scaled_far_point_factor = 1.0 + (far_point_factor - 1.0) * stack_scale
        contour = bridged_smoothed[idx]
        inflated = inflate_contour(contour, scaled_inflation, scaled_far_point_factor)
        inflation_scaled_contours.append(inflated)

    for idx_i, idx in enumerate(contour_indices):
        contour = inflation_scaled_contours[idx_i]
        color = colors[idx]
        hex_color = svgwrite.rgb(*color)
        path_data = contour_to_svg_path(contour)
        dwg.add(dwg.path(d=path_data, fill=hex_color))
        report(85 + int(10 * (idx_i+1) / len(contour_indices)))

    dwg.save()
    report(100)
    print(f"[SVG saved] {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bridged and smoothed vector blobs from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="directional_bridged_blobs.svg", help="Output SVG path")
    parser.add_argument("--resize", type=int, default=None, help="Resize width (maintain aspect ratio)")
    parser.add_argument("--segments", type=int, default=40, help="Number of SLIC segments")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness")
    parser.add_argument("--max-colors", type=int, default=8, help="Max number of quantized colors")
    parser.add_argument("--bridge-distance", type=float, default=5.0, help="Bridge distance")
    parser.add_argument("--color-tolerance", type=float, default=10.0, help="LAB color tolerance")
    parser.add_argument("--proximity-threshold", type=float, default=50.0, help="Proximity threshold")
    parser.add_argument("--falloff-radius", type=int, default=5, help="Falloff radius for bridge influence")
    parser.add_argument("--max-curvature", type=float, default=160.0, help="Maximum curvature angle in degrees")
    parser.add_argument("--smooth-iterations", type=int, default=3, help="Number of smoothing iterations")
    parser.add_argument("--smooth-alpha", type=float, default=0.3, help="Smoothing strength (alpha)")
    parser.add_argument("--shape-order", type=str, default="area", choices=["area", "brightness", "position_x", "position_y", "position_centre"], help="Ordering method for stacking SVG blobs (area, brightness)")

    args = parser.parse_args()
    image = load_image(args.image, resize_width=args.resize)

    create_directional_bridged_svg(
        image,
        output_path=args.output,
        n_segments=args.segments,
        compactness=args.compactness,
        max_colors=args.max_colors,
        bridge_distance=args.bridge_distance,
        color_tolerance=args.color_tolerance,
        proximity_threshold=args.proximity_threshold,
        falloff_radius=args.falloff_radius,
        max_curvature=np.radians(args.max_curvature),
        smooth_iterations=args.smooth_iterations,
        smooth_alpha=args.smooth_alpha,
        shape_order=args.shape_order,
        inflation_amount=args.inflation_amount
    )