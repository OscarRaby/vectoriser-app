# import streamlit as st
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
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
import threading

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

# --- Kivy GUI Code ---
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
import threading

class VectoriserBox(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.img_np = None
        self.pil_img = None
        self.image_widget = KivyImage(size_hint=(1, 0.5))
        self.add_widget(self.image_widget)
        self.file_chooser = FileChooserIconView(size_hint=(1, 0.3))
        self.file_chooser.filters = ['*.png', '*.jpg', '*.jpeg']
        self.add_widget(self.file_chooser)
        select_btn = Button(text='Select Image', size_hint=(1, None), height=40)
        select_btn.bind(on_release=self.on_select_image)
        self.add_widget(select_btn)
        # Parameters
        self.param_widgets = {}
        self.param_values = {
            'Noise Scale': 60.0,
            'Blur Sigma': 2.0,
            'Compactness': 0.001,
            'Max Colors': 8,
            'Bridge Distance': 5.0,
            'Color Tolerance': 10.0,
            'Proximity Threshold': 50.0,
            'Falloff Radius': 5,
            'Max Curvature': 160.0,
            'Smooth Iterations': 3,
            'Smooth Alpha': 0.3,
            'Blob Inflation Amount': 0.0,
            'Far Point Inflation Factor': 1.0,
        }
        self.modifier_values = {
            'Color Quantization': True,
            'Bridging': True,
            'Smoothing': True,
            'Inflation': True,
        }
        self.param_sliders = {}
        for key, val in self.param_values.items():
            box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=30)
            box.add_widget(Label(text=key, size_hint=(0.5, 1)))
            if isinstance(val, float) or isinstance(val, int):
                slider = Slider(min=0, max=200 if 'Scale' in key or 'Distance' in key or 'Amount' in key else 10, value=val, step=0.01 if isinstance(val, float) else 1)
                slider.bind(value=self.on_slider_change(key))
                self.param_sliders[key] = slider
                box.add_widget(slider)
            self.add_widget(box)
        # Modifiers
        for key, val in self.modifier_values.items():
            box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=30)
            box.add_widget(Label(text=key, size_hint=(0.7, 1)))
            cb = CheckBox(active=val)
            cb.bind(active=self.on_checkbox_change(key))
            box.add_widget(cb)
            self.add_widget(box)
        # Run button
        self.run_btn = Button(text='Run Vectoriser', size_hint=(1, None), height=40)
        self.run_btn.bind(on_release=self.run_vectoriser)
        self.add_widget(self.run_btn)
        # Output image
        self.output_image = KivyImage(size_hint=(1, 0.5))
        self.add_widget(self.output_image)
        # Progress bar
        self.progress = ProgressBar(max=100, value=0, size_hint=(1, None), height=20)
        self.add_widget(self.progress)
        # Status label
        self.status_label = Label(text='Ready.', size_hint=(1, None), height=30)
        self.add_widget(self.status_label)

    def on_slider_change(self, key):
        def _on_value(instance, value):
            self.param_values[key] = value
        return _on_value

    def on_checkbox_change(self, key):
        def _on_active(instance, value):
            self.modifier_values[key] = value
        return _on_active

    def on_select_image(self, instance):
        selection = self.file_chooser.selection
        if selection:
            path = selection[0]
            self.pil_img = Image.open(path).convert('RGB')
            self.pil_img = self.pil_img.resize((360, 240), Image.LANCZOS)
            self.img_np = img_as_float(np.array(self.pil_img))
            self.update_image_widget(self.pil_img, self.image_widget)
            self.status_label.text = f"Loaded: {path}"

    def load_image(self, instance, selection):
        if selection:
            path = selection[0]
            self.pil_img = Image.open(path).convert('RGB')
            self.pil_img = self.pil_img.resize((360, 240), Image.LANCZOS)
            self.img_np = img_as_float(np.array(self.pil_img))
            self.update_image_widget(self.pil_img, self.image_widget)
            self.status_label.text = f"Loaded: {path}"

    def update_image_widget(self, pil_img, widget):
        arr = np.array(pil_img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr.astype(np.uint8)
            buf = arr.tobytes()
            texture = Texture.create(size=(arr.shape[1], arr.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            widget.texture = texture
            widget.canvas.ask_update()
        elif arr.ndim == 2:  # grayscale
            arr = np.stack([arr]*3, axis=-1).astype(np.uint8)
            buf = arr.tobytes()
            texture = Texture.create(size=(arr.shape[1], arr.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            widget.texture = texture
            widget.canvas.ask_update()

    def run_vectoriser(self, *args):
        if self.img_np is None:
            self.status_label.text = "No image loaded."
            return
        self.status_label.text = "Running..."
        self.progress.value = 0
        threading.Thread(target=self._run_vectoriser_backend).start()

    def _run_vectoriser_backend(self):
        # --- Noise Watershed Segmentation ---
        labels = noise_watershed(self.img_np, scale=self.param_values['Noise Scale'], blur_sigma=self.param_values['Blur Sigma'], compactness=self.param_values['Compactness'])
        self.progress.value = 10
        # --- Color Quantization ---
        if self.modifier_values['Color Quantization']:
            quantized_image, _ = quantize_image_colors((self.img_np * 255).astype(np.uint8), int(self.param_values['Max Colors']))
        else:
            quantized_image = (self.img_np * 255).astype(np.uint8)
        self.progress.value = 20
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
        self.progress.value = 40
        # --- Bridging ---
        if self.modifier_values['Bridging']:
            bridged = improved_bridge_contours(
                contours, centroids, colors,
                bridge_distance=self.param_values['Bridge Distance'],
                color_tolerance=self.param_values['Color Tolerance'],
                proximity_threshold=self.param_values['Proximity Threshold'],
                falloff_radius=int(self.param_values['Falloff Radius']),
                max_curvature=np.radians(self.param_values['Max Curvature'])
            )
        else:
            bridged = contours
        self.progress.value = 60
        # --- Smoothing ---
        if self.modifier_values['Smoothing']:
            bridged_smoothed = [
                laplacian_smooth(c, iterations=int(self.param_values['Smooth Iterations']), alpha=self.param_values['Smooth Alpha'])
                for c in bridged
            ]
        else:
            bridged_smoothed = bridged
        self.progress.value = 75
        # --- Inflation ---
        inflation_scaled_contours = []
        for contour in bridged_smoothed:
            if self.modifier_values['Inflation']:
                inflated = inflate_contour(contour, self.param_values['Blob Inflation Amount'], self.param_values['Far Point Inflation Factor'])
            else:
                inflated = contour
            inflation_scaled_contours.append(inflated)
        self.progress.value = 90
        # --- SVG Generation ---
        h, w = self.img_np.shape[:2]
        dwg = svgwrite.Drawing(size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
        dwg.defs.add(dwg.style("path { stroke: none; stroke-width: 0; }"))
        for contour, color_val in zip(inflation_scaled_contours, colors):
            hex_color = svgwrite.rgb(*color_val)
            path_data = contour_to_svg_path(contour)
            dwg.add(dwg.path(d=path_data, fill=hex_color))
        svg_string = dwg.tostring()
        # --- Show Output ---
        png_bytes = cairosvg.svg2png(bytestring=svg_string.encode("utf-8"), output_width=w, output_height=h)
        output_img = Image.open(io.BytesIO(png_bytes))
        self.update_image_widget(output_img, self.output_image)
        self.progress.value = 100
        self.status_label.text = "Done!"

class OrganicSegmentationKivyApp(App):
    def build(self):
        Window.size = (900, 900)
        return VectoriserBox()

if __name__ == "__main__":
    OrganicSegmentationKivyApp().run()