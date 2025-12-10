import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.measure import find_contours
from scipy.spatial import KDTree
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
import svgwrite
import os
import io
import cairosvg
import json

def load_image(image_path, resize_width=None):
    image = Image.open(image_path).convert('RGB')
    if resize_width:
        w_percent = resize_width / float(image.width)
        new_height = int(float(image.height) * w_percent)
        image = image.resize((resize_width, new_height), Image.LANCZOS)
    return np.array(image), image

def quantize_image_colors(image, max_colors):
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

def improved_bridge_contours(contours, centroids, colors, bridge_distance,
                              color_tolerance, proximity_threshold, falloff_radius, max_curvature):
    lab_colors = rgb2lab(np.array(colors).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    tree = KDTree(centroids)
    bridged = []
    for i, contour in enumerate(contours):
        current_contour = np.copy(contour)
        indices = tree.query_ball_point(centroids[i], r=proximity_threshold)
        for j in indices:
            if j == i: continue
            if np.linalg.norm(lab_colors[i] - lab_colors[j]) >= color_tolerance: continue
            dists = np.linalg.norm(current_contour - centroids[j], axis=1)
            idx = np.argmin(dists)
            target = contours[j]
            cp = target[np.argmin(np.linalg.norm(target - current_contour[idx], axis=1))]
            direction = cp - current_contour[idx]
            norm = np.linalg.norm(direction)
            if norm == 0: continue
            direction /= norm
            n = len(current_contour)
            for offset in range(-falloff_radius, falloff_radius+1):
                ni = (idx+offset) % n
                p = (ni-1) % n; nx = (ni+1)%n
                v1 = current_contour[ni]-current_contour[p]
                v2 = current_contour[nx]-current_contour[ni]
                a = np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),-1,1))
                if a>max_curvature: continue
                w = 0.5*(1+np.cos(np.pi*offset/falloff_radius))
                current_contour[ni] += direction*bridge_distance*w
        bridged.append(current_contour)
    return bridged

def generate_svg(params, progress_callback=None):
    resize_width = params.get('resize_width')
    img_np, img_pil = load_image(params['image'], resize_width)
    qi, _ = quantize_image_colors(img_np, params['max_colors'])
    seg = slic(img_as_float(qi), n_segments=params['segments'], compactness=params['compactness'], start_label=1)
    dwg = svgwrite.Drawing(params['output'], size=(f"{img_np.shape[1]}px",f"{img_np.shape[0]}px"), viewBox=f"0 0 {img_np.shape[1]} {img_np.shape[0]}")
    dwg.defs.add(dwg.style("path{stroke:none;stroke-width:0} "))
    contours, cents, cols = [], [], []
    total = len(np.unique(seg))
    for idx, sid in enumerate(np.unique(seg)):
        mask = seg==sid; avg = np.mean(qi[mask],axis=0).astype(int)
        f = find_contours(mask.astype(float),0.5)
        if not f: continue
        c = f[0]; contours.append(c); cents.append(np.mean(c,axis=0)); cols.append(avg)
    if progress_callback:
        progress_callback(60)

    bridged = improved_bridge_contours(contours,cents,cols,
                                       params['bridge_distance'],params['color_tolerance'],
                                       params['proximity_threshold'],params['falloff_radius'],
                                       np.radians(params['max_curvature']))
    if progress_callback:
        progress_callback(70)

    sm = [laplacian_smooth(c,params['smooth_iterations'],params['smooth_alpha']) for c in bridged]
    for i, (c, col) in enumerate(zip(sm, cols)):
        dwg.add(dwg.path(d=contour_to_svg_path(c), fill=svgwrite.rgb(*col)))
        if progress_callback:
            progress_callback(70 + int(25 * i / len(sm)), f"Processed shape {i + 1} of {len(sm)}")
        if progress_callback:
            progress_callback(70 + int(25 * i / len(sm)))
    dwg.save()
    return img_pil, dwg.tostring()

# GUI
root = tk.Tk(); root.title('Bridged Blob Vectoriser')

import json

PRESETS_FILE = 'vectoriser_presets.json'
LAST_USED_FILE = 'last_used_preset.txt'
try:
    with open(PRESETS_FILE, 'r') as f:
        presets = json.load(f)
except FileNotFoundError:
    presets = {}

def save_preset():
    name = preset_name.get()
    if name:
        presets[name] = {k: v.get() for k, v in entries.items()}
        with open(PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=2)
        preset_menu['menu'].add_command(label=name, command=lambda v=name: load_preset(v))
        messagebox.showinfo('Preset Saved', f'Preset "{name}" saved.')

def load_preset(name):
    with open(LAST_USED_FILE, 'w') as f:
        f.write(name)
    for k, v in presets[name].items():
        entries[k].set(v)

def update_preset(name):
    if name in presets:
        presets[name] = {k: v.get() for k, v in entries.items()}
        with open(PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=2)
        messagebox.showinfo('Preset Updated', f'Preset "{name}" updated.')
    else:
        messagebox.showerror('Error', f'Preset "{name}" does not exist.')

def delete_preset(name):
    if name in presets:
        del presets[name]
        with open(PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=2)
        preset_menu['menu'].delete(0, 'end')
        for p in presets:
            preset_menu['menu'].add_command(label=p, command=lambda v=p: load_preset(v))
        preset_var.set('')
        messagebox.showinfo('Preset Deleted', f'Preset "{name}" deleted.')
    else:
        messagebox.showerror('Error', f'Preset "{name}" does not exist.')

preset_name = tk.StringVar()
tk.Label(root, text='Preset Name').grid(row=14, column=0, sticky='e')
tk.Entry(root, textvariable=preset_name, width=10).grid(row=14, column=1)
tk.Button(root, text='Save Preset', command=save_preset).grid(row=14, column=2)
tk.Button(root, text='Update Preset', command=lambda: update_preset(preset_var.get())).grid(row=14, column=4)
tk.Button(root, text='Delete Preset', command=lambda: delete_preset(preset_var.get())).grid(row=14, column=5)

try:
    with open(LAST_USED_FILE, 'r') as f:
        last_used = f.read().strip()
except FileNotFoundError:
    last_used = next(iter(presets), '')

preset_var = tk.StringVar(value=last_used)
preset_menu = tk.OptionMenu(root, preset_var, last_used, *presets.keys(), command=load_preset)
preset_menu.grid(row=14, column=3)
# Load preset after all entries are created
entries = {}
def make_entry(label, row, default):
    tk.Label(root, text=label).grid(row=row,column=0,sticky='e')
    var = tk.StringVar(value=str(default))
    entry = tk.Entry(root, textvariable=var, width=10)
    entry.grid(row=row,column=1)
    entries[label]=var
    return var

def select_image():
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if filename:
        entries['Image Path'].set(filename)
        preview_image(filename)

def select_output():
    filename = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files", "*.svg")])
    if filename:
        entries['Output SVG'].set(filename)

def preview_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img.thumbnail((300, 300))
        preview_img = ImageTk.PhotoImage(img)
        preview_label.config(image=preview_img)
        preview_label.image = preview_img
    except Exception as e:
        messagebox.showerror('Preview Error', str(e))

def preview_svg(svg_string):
    try:
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        img = Image.open(io.BytesIO(png_data))
        img.thumbnail((300, 300))
        preview_img = ImageTk.PhotoImage(img)
        svg_preview_label.config(image=preview_img)
        svg_preview_label.image = preview_img
    except Exception as e:
        messagebox.showerror('SVG Preview Error', str(e))

make_entry('Image Path',0,'')
tk.Button(root, text='Browse', command=select_image).grid(row=0,column=2)
make_entry('Output SVG',1,'out.svg')
tk.Button(root, text='Save As', command=select_output).grid(row=1,column=2)
make_entry('Resize Width',2,'None')
make_entry('Segments',3,'40')
make_entry('Compactness',4,'10')
make_entry('Max Colors',5,'8')
make_entry('Bridge Distance',6,'5')
make_entry('Color Tolerance',7,'10')
make_entry('Proximity Threshold',8,'50')
make_entry('Falloff Radius',9,'5')
make_entry('Max Curvature',10,'160')
make_entry('Smooth Iterations',11,'3')
make_entry('Smooth Alpha',12,'0.3')

# Now that entries are defined, load the preset
if preset_var.get():
    load_preset(preset_var.get())
    preset_name.set(preset_var.get())

preview_label = tk.Label(root)
preview_label.grid(row=0, column=3, rowspan=10, padx=10, pady=10)
svg_preview_label = tk.Label(root)
svg_preview_label.grid(row=0, column=4, rowspan=10, padx=10, pady=10)

import time

def run():
    progress['value'] = 0
    progress.update_idletasks()
    root.update_idletasks()
    try:
        progress['value'] = 10
        progress.update_idletasks()
        progress['value'] = 20
        progress.update_idletasks()
        params = {k.replace(' ','_').lower(): (None if v.get()=='None' else float(v.get()) if '.' in v.get() else int(v.get())) for k,v in entries.items() if k not in ['Image Path', 'Output SVG']}
        params['image'] = entries['Image Path'].get()
        params['output'] = entries['Output SVG'].get()
        progress['value'] = 30
        progress.update_idletasks()
        progress['value'] = 50
        progress.update_idletasks()
        def update_progress(value, message=None):
            progress['value'] = value
            progress.update_idletasks()
            if message:
                progress_label.config(text=message)
            else:
                pass  # Avoid flickering by not clearing the label unnecessarily

        start_time = time.time()
        image, svg_data = generate_svg(params, progress_callback=update_progress)
        duration = time.time() - start_time
        progress['value'] = 80
        progress.update_idletasks()
        progress['value'] = 90
        progress.update_idletasks()
        preview_image(params['image'])
        preview_svg(svg_data)
        progress['value'] = 100
        progress.update_idletasks()
        progress.stop()
        summary = f"SVG generated successfully. | Total shapes: {len(svg_data.split('<path')) - 1} | Time: {duration:.2f} sec"
        progress_label.config(text=summary)
    except Exception as e:
        progress.stop()
        messagebox.showerror('Error', str(e))

progress = ttk.Progressbar(root, mode='determinate', maximum=100)
progress.grid(row=13, column=0, columnspan=4, sticky='we', pady=5)
progress_label = tk.Label(root, text="")
progress_label.grid(row=13, column=5, sticky='w')
tk.Button(root, text='Generate SVG', command=run).grid(row=13, column=4)
root.mainloop()

