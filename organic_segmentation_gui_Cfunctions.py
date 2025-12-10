import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import color
from skimage.util import img_as_float
import io
import cairosvg
from directional_bridged_blob_vectoriser import (
    load_image, quantize_image_colors, improved_bridge_contours, laplacian_smooth,
    inflate_contour, contour_to_svg_path
)
from skimage.measure import find_contours
import svgwrite

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
        }
        # --- GUI layout ---
        tk.Button(root, text="Load Image", command=self.load_image).grid(row=0, column=0, columnspan=2, pady=5)
        row = 1
        for key in [
            "Noise Scale", "Blur Sigma", "Compactness", "Max Colors", "Bridge Distance", "Color Tolerance",
            "Proximity Threshold", "Falloff Radius", "Max Curvature", "Smooth Iterations", "Smooth Alpha",
            "Blob Inflation Amount", "Far Point Inflation Factor"]:
            tk.Label(root, text=key).grid(row=row, column=0, sticky='e')
            # Use sliders for parameter entry
            if key == "Noise Scale":
                tk.Scale(root, variable=self.param_vars[key], from_=-100.0, to=1000.0, resolution=10.0, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Blur Sigma":
                tk.Scale(root, variable=self.param_vars[key], from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL).grid(row=row, column=1)
            elif key == "Compactness":
                tk.Scale(root, variable=self.param_vars[key], from_=0.0001, to=0.1, resolution=0.0001, orient=tk.HORIZONTAL).grid(row=row, column=1)  # Avoid zero
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
        tk.Button(root, text="Run Vectoriser", command=self.run_vectoriser).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        # Modifier switches
        self.modifier_vars = {
            "Color Quantization": tk.BooleanVar(value=True),
            "Bridging": tk.BooleanVar(value=True),
            "Smoothing": tk.BooleanVar(value=True),
            "Inflation": tk.BooleanVar(value=True),
        }
        for mod_key in self.modifier_vars:
            tk.Checkbutton(root, text=f"Enable {mod_key}", variable=self.modifier_vars[mod_key]).grid(row=row, column=0, columnspan=2, sticky='w')
            row += 1
        # Scaling method selector for segmentation parameters
        self.scaling_method = tk.StringVar(value="max")
        tk.Label(root, text="Segmentation Scaling").grid(row=row, column=0, sticky='e')
        tk.OptionMenu(root, self.scaling_method, "max", "min", "average", "area", "sqrt_area").grid(row=row, column=1)
        row += 1
        tk.Label(root, text="Segmentation Parameter Divider").grid(row=row, column=0, sticky='e')
        tk.Scale(root, variable=self.param_vars["Segmentation Multiplier"], from_=-10.0, to=10.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=row, column=1)
        row += 1
        # Progress bar (move below Run button)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = tk.Scale(root, variable=self.progress_var, from_=0, to=100, orient=tk.HORIZONTAL, length=300, showvalue=0)
        self.progress_bar.grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor='w', relief=tk.SUNKEN, bd=1)
        self.status_label.grid(row=row, column=0, columnspan=2, sticky='we')
        row +=1
        # Matplotlib output (restore visualisation box)
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=row)
        # Place preview controls and output immediately below Matplotlib output in the same column
        preview_row = row
        self.preview_fig, self.preview_ax = plt.subplots(figsize=(2, 2), dpi=75)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=root)
        self.preview_canvas.get_tk_widget().grid(row=preview_row, column=2, sticky='n')
        preview_row += 1
        tk.Button(root, text="Preview Parameters", command=self.run_preview).grid(row=preview_row, column=2, sticky='ew', pady=2)
        preview_row += 1
        # Initialize preview_mode before using it
        self.preview_mode = tk.StringVar(value="manual")
        tk.OptionMenu(root, self.preview_mode, "manual", "auto").grid(row=preview_row, column=2, sticky='ew')
        preview_row += 1
        # Bind auto preview
        for var in list(self.param_vars.values()) + list(self.modifier_vars.values()):
            var.trace_add('write', lambda *args: self.run_preview() if self.preview_mode.get() == "auto" else None)
        # Preview controls
        self.preview_mode = tk.StringVar(value="manual")

        # --- Preset controls ---
        self.preset_name_var = tk.StringVar()
        self.preset_menu_var = tk.StringVar()
        self.presets_file = 'vectoriser_presets.json'
        import json, os, io
        try:
            with open(self.presets_file, 'r') as f:
                self.presets = json.load(f)
        except Exception:
            self.presets = {}
        self.PRESET_PARAMS = [k for k in self.param_vars if k not in ("SVG Output Path", "Stacking Order")]
        # Preset save/load/update/delete
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
        tk.Label(root, text="Preset Name").grid(row=row, column=0, sticky='e', pady=(10,2))
        tk.Entry(root, textvariable=self.preset_name_var).grid(row=row, column=1, pady=(10,2))
        tk.Button(root, text="Save Preset", command=save_preset).grid(row=row, column=1, pady=(10,2))
        row += 1
        self.preset_menu = tk.OptionMenu(root, self.preset_menu_var, *self.presets.keys(), command=load_preset)
        self.preset_menu.grid(row=row, column=0, pady=2)
        tk.Button(root, text="Update Preset", command=update_preset).grid(row=row, column=1, pady=2)
        tk.Button(root, text="Delete Preset", command=delete_preset).grid(row=row, column=1, pady=2)
        row += 1
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
        tk.Button(root, text="Undo", command=undo).grid(row=row, column=0, sticky='ew')
        tk.Button(root, text="Redo", command=redo).grid(row=row, column=1, sticky='ew')
        row += 1
        # Save history on parameter change
        for var in list(self.param_vars.values()) + list(self.modifier_vars.values()):
            var.trace_add('write', lambda *args: save_history())

    def load_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if filename:
            img = Image.open(filename).convert('RGB')
            # Maintain aspect ratio, fit within max size (e.g. 360x240)
            max_w, max_h = 360, 240
            w, h = img.size
            scale = min(max_w / w, max_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
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
        if self.img_np is None:
            return
        self.update_progress(5, "Starting segmentation...")
        crop_shape = (150, 150)
        scale_factor = self.get_scale_factor(self.img_np.shape, crop_shape=crop_shape)
        multiplier = self.param_vars["Segmentation Multiplier"].get()
        # Failsafe: avoid divide by zero
        if multiplier == 0:
            multiplier = 1e-6
        noise_scale = self.param_vars["Noise Scale"].get() * scale_factor / multiplier
        blur_sigma = self.param_vars["Blur Sigma"].get() * scale_factor / multiplier
        compactness = self.param_vars["Compactness"].get() * scale_factor / multiplier
        if compactness == 0:
            compactness = 1e-6
        if blur_sigma == 0:
            blur_sigma = 1e-6
        if noise_scale == 0:
            noise_scale = 1e-6
        self.status_var.set(
            f"Scaled parameters: Noise Scale={noise_scale:.2f}, Blur Sigma={blur_sigma:.2f}, Compactness={compactness:.6f}, Multiplier={multiplier:.2f}"
        )
        self.root.update_idletasks()
        labels = noise_watershed(self.img_np,
            scale=noise_scale,
            blur_sigma=blur_sigma,
            compactness=compactness)
        self.update_progress(20, "Segmentation complete. Running color quantization...")
        # Color quantization (modifier)
        if self.modifier_vars["Color Quantization"].get():
            quantized_image, _ = quantize_image_colors((self.img_np*255).astype(np.uint8), self.param_vars["Max Colors"].get())
        else:
            quantized_image = (self.img_np*255).astype(np.uint8)
        self.update_progress(30, "Color quantization complete. Extracting contours...")
        # Find contours and main color for each label
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
        self.update_progress(40, "Contours extracted. Running bridging...")
        # Bridging (modifier)
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
        self.update_progress(55, "Bridging complete. Running smoothing...")
        # Smoothing (modifier)
        if self.modifier_vars["Smoothing"].get():
            bridged_smoothed = [
                laplacian_smooth(c, iterations=self.param_vars["Smooth Iterations"].get(), alpha=self.param_vars["Smooth Alpha"].get())
                for c in bridged
            ]
        else:
            bridged_smoothed = bridged
        self.update_progress(65, "Smoothing complete. Running inflation...")
        # Inflation and stacking order (modifier)
        def contour_area(contour):
            x = contour[:, 1]
            y = contour[:, 0]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        def luminance(rgb):
            r, g, b = rgb
            return 0.2126*r + 0.7152*g + 0.0722*b
        h, w = self.img_np.shape[:2]
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
        # Inflation scaling
        N = len(bridged_smoothed)
        inflation_scaled_contours = []
        inflation_stacking = self.param_vars["Inflation Proportional to Stacking"].get()
        inflation_amount = self.param_vars["Blob Inflation Amount"].get()
        far_point_factor = self.param_vars["Far Point Inflation Factor"].get()
        for order_idx, idx in enumerate(contour_indices):
            contour = bridged_smoothed[idx]
            if self.modifier_vars["Inflation"].get():
                if inflation_stacking:
                    if N > 1:
                        stack_scale = np.log1p(order_idx) / np.log1p(N-1)
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
        self.update_progress(80, "Inflation complete. Generating SVG...")
        # SVG generation
        dwg = svgwrite.Drawing(filename=self.svg_path.get(), size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
        dwg.defs.add(dwg.style("path { stroke: none; stroke-width: 0; }"))
        for idx_i, idx in enumerate(contour_indices):
            contour = inflation_scaled_contours[idx_i]
            color_val = colors[idx]
            hex_color = svgwrite.rgb(*color_val)
            path_data = contour_to_svg_path(contour)
            dwg.add(dwg.path(d=path_data, fill=hex_color))
        dwg.save()
        self.update_progress(100, "SVG file saved. Vectorisation complete!")
        # Show SVG preview
        self.show_svg_preview(self.svg_path.get())
        self.update_progress(0, "Ready.")

    def show_image(self, img_np):
        self.ax.clear()
        if img_np.ndim == 2:
            self.ax.imshow(img_np, cmap='gray')
        else:
            self.ax.imshow(img_np)
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()

    def show_svg_preview(self, svg_path):
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_string = f.read()
        # Use the actual image size for preview
        w, h = self.image.size if self.image else (360, 240)
        png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), output_width=w, output_height=h)
        img = Image.open(io.BytesIO(png_bytes))
        self.show_image(np.array(img))

    def run_preview(self):
        if self.img_np is None:
            return
        # Crop a 150x150 section from the center
        h, w = self.img_np.shape[:2]
        ch, cw = 150, 150
        y0 = max(0, h//2 - ch//2)
        x0 = max(0, w//2 - cw//2)
        y1 = min(h, y0 + ch)
        x1 = min(w, x0 + cw)
        img_crop = self.img_np[y0:y1, x0:x1]
        # Failsafe: avoid divide by zero for preview parameters
        noise_scale = self.param_vars["Noise Scale"].get()
        blur_sigma = self.param_vars["Blur Sigma"].get()
        compactness = self.param_vars["Compactness"].get()
        if compactness == 0:
            compactness = 1e-6
        if blur_sigma == 0:
            blur_sigma = 1e-6
        if noise_scale == 0:
            noise_scale = 1e-6
        labels = noise_watershed(img_crop,
            scale=noise_scale,
            blur_sigma=blur_sigma,
            compactness=compactness)
        if self.modifier_vars["Color Quantization"].get():
            quantized_image, _ = quantize_image_colors((img_crop*255).astype(np.uint8), self.param_vars["Max Colors"].get())
        else:
            quantized_image = (img_crop*255).astype(np.uint8)
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
        if self.modifier_vars["Smoothing"].get():
            bridged_smoothed = [
                laplacian_smooth(c, iterations=self.param_vars["Smooth Iterations"].get(), alpha=self.param_vars["Smooth Alpha"].get())
                for c in bridged
            ]
        else:
            bridged_smoothed = bridged
        N = len(bridged_smoothed)
        inflation_stacking = self.param_vars["Inflation Proportional to Stacking"].get()
        inflation_amount = self.param_vars["Blob Inflation Amount"].get()
        far_point_factor = self.param_vars["Far Point Inflation Factor"].get()
        inflation_scaled_contours = []
        for order_idx, contour in enumerate(bridged_smoothed):
            if self.modifier_vars["Inflation"].get():
                if inflation_stacking:
                    if N > 1:
                        stack_scale = np.log1p(order_idx) / np.log1p(N-1)
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
        # Render preview
        self.preview_ax.clear()
        self.preview_ax.imshow(img_crop)
        for i, contour in enumerate(inflation_scaled_contours):
            color_val = colors[i]
            self.preview_ax.plot(contour[:,1], contour[:,0], color=np.array(color_val)/255, linewidth=2)
        self.preview_ax.axis('off')
        self.preview_fig.tight_layout()
        self.preview_canvas.draw()

        # Calculate scaling factor based on area
        full_h, full_w = self.img_np.shape[:2]
        crop_h, crop_w = 150, 150
        area_scale = (full_h * full_w) / (crop_h * crop_w)
        sqrt_area_scale = np.sqrt(area_scale)

if __name__ == "__main__":
    root = tk.Tk()
    app = OrganicSegmentationGUI(root)
    root.mainloop()