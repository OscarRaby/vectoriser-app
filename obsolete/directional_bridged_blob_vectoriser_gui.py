import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from theme_loader import GUITheme
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import io
import cairosvg
import json
import os
import threading
import time

from skimage.segmentation import slic
from skimage.util import img_as_float

from directional_bridged_blob_vectoriser import (
    load_image, create_directional_bridged_svg
)

theme = GUITheme()
root = tk.Tk()
root.title("Directional Bridged Blob Vectoriser")
if theme["borderless"]:
    root.overrideredirect(True)
root.configure(bg=theme["background"])
default_font = theme.font()

THUMBS_DIR = os.path.join(os.path.dirname(__file__), "preset_thumbs")
os.makedirs(THUMBS_DIR, exist_ok=True)

PRESETS_FILE = 'vectoriser_presets.json'
try:
    with open(PRESETS_FILE, 'r') as f:
        presets = json.load(f)
except Exception:
    presets = {}

# --- Param Variables ---
preset_name_var = tk.StringVar()
status_var = tk.StringVar(value="")
stack_order_var = tk.StringVar(value="area")

param_vars = {
    "Image Path": tk.StringVar(),
    "Output SVG": tk.StringVar(value="out.svg"),
    "Resize Width": tk.StringVar(value="None"),
    "Segments": tk.StringVar(value="40"),
    "Compactness": tk.StringVar(value="10.0"),
    "Max Colors": tk.StringVar(value="8"),
    "Bridge Distance": tk.StringVar(value="5.0"),
    "Color Tolerance": tk.StringVar(value="10.0"),
    "Proximity Threshold": tk.StringVar(value="50.0"),
    "Falloff Radius": tk.StringVar(value="5"),
    "Max Curvature": tk.StringVar(value="160.0"),
    "Smooth Iterations": tk.StringVar(value="3"),
    "Smooth Alpha": tk.StringVar(value="0.3"),
    "Stacking Order": stack_order_var,
}
param_vars["Enable Grid-like Blob Merging"] = tk.BooleanVar(value=False)
param_vars["Grid Area Threshold"] = tk.StringVar(value="7000")
param_vars["Grid Compactness Threshold"] = tk.StringVar(value="1.35")
param_vars["Blob Inflation Amount"] = tk.StringVar(value="0.0")
param_vars["Far Point Inflation Factor"] = tk.StringVar(value="1.0")
param_vars["Inflation Proportional to Stacking"] = tk.BooleanVar(value=True)

parameter_history = []
history_index = -1  # Start before first entry

PRESET_PARAMS = [k for k in param_vars if k not in ("Image Path", "Output SVG")]

STACK_ORDER_OPTIONS = [
    ("Area (largest behind)", "area"),
    ("Area (smallest behind)", "area_reverse"),
    ("Brightness (darkest behind)", "brightness"),
    ("Brightness (brightest behind)", "brightness_reverse"),
    ("Position Y (top to bottom)", "position_y"),
    ("Position Y (bottom to top)", "position_y_reverse"),
    ("Position X (left to right)", "position_x"),
    ("Position X (right to left)", "position_x_reverse"),
    ("Position Centre (outward-in)", "position_centre"),
    ("Position Centre (inward-out)", "position_centre_reverse")
]
stack_order_label_to_key = dict(STACK_ORDER_OPTIONS)
stack_order_key_to_label = {v: k for k, v in stack_order_label_to_key.items()}

# --- Themed helpers ---
def themed_label(parent, text): return tk.Label(parent, text=text, fg=theme["label_fg"], bg=theme["background"], font=default_font)
def themed_entry(parent, var): return tk.Entry(parent, textvariable=var, bg=theme["input_bg"], fg=theme["input_fg"], font=default_font, insertbackground=theme["input_fg"])
def themed_button(parent, text, command): return tk.Button(parent, text=text, command=command, bg=theme["button_bg"], fg=theme["button_fg"], font=default_font, activebackground=theme["highlight"])

# --- GUI Layout ---
param_frame = tk.Frame(root, bg=theme["background"])
param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
preview_frame = tk.Frame(root, bg=theme["background"])
preview_frame.grid(row=0, column=1, rowspan=20, padx=10, pady=10)

# --- Parameter entries ---
row = 0
for key in ["Image Path", "Output SVG", "Resize Width", "Segments", "Compactness", "Max Colors", "Bridge Distance", "Color Tolerance", "Proximity Threshold", "Falloff Radius", "Max Curvature", "Smooth Iterations", "Smooth Alpha"]:
    themed_label(param_frame, key).grid(row=row, column=0, sticky="e", pady=2)
    themed_entry(param_frame, param_vars[key]).grid(row=row, column=1, pady=2)
    if key == "Image Path":
        themed_button(param_frame, "Browse", lambda: select_image()).grid(row=row, column=2, pady=2)
    if key == "Output SVG":
        themed_button(param_frame, "Save As", lambda: select_output()).grid(row=row, column=2, pady=2)
    row += 1

themed_label(param_frame, "Enable Grid-like Blob Merging").grid(row=row, column=0, sticky="e", pady=2)
tk.Checkbutton(param_frame, variable=param_vars["Enable Grid-like Blob Merging"], bg=theme["background"]).grid(row=row, column=1, pady=2)
row += 1

themed_label(param_frame, "Grid Area Threshold").grid(row=row, column=0, sticky="e", pady=2)
themed_entry(param_frame, param_vars["Grid Area Threshold"]).grid(row=row, column=1, pady=2)
row += 1

themed_label(param_frame, "Grid Compactness Threshold").grid(row=row, column=0, sticky="e", pady=2)
themed_entry(param_frame, param_vars["Grid Compactness Threshold"]).grid(row=row, column=1, pady=2)
row += 1

themed_label(param_frame, "Blob Inflation Amount").grid(row=row, column=0, sticky="e")
themed_entry(param_frame, param_vars["Blob Inflation Amount"]).grid(row=row, column=1)
row += 1

themed_label(param_frame, "Far Point Inflation Factor").grid(row=row, column=0, sticky="e")
themed_entry(param_frame, param_vars["Far Point Inflation Factor"]).grid(row=row, column=1)
row += 1

stacking_switch_label = themed_label(param_frame, "Inflation Proportional to Stacking Order")
stacking_switch_label.grid(row=row, column=0, sticky="e")
tk.Checkbutton(param_frame, variable=param_vars["Inflation Proportional to Stacking"], bg=theme["background"]).grid(row=row, column=1)
row += 1

# --- Preset controls ---
def save_preset():
    name = preset_name_var.get().strip()
    if not name:
        messagebox.showerror("Preset", "Enter a preset name.")
        return
    preset_dict = {}
    for k, v in param_vars.items():
        if k in PRESET_PARAMS:
            if k == "Stacking Order":
                # Save the internal key, not the label
                preset_dict[k] = stack_order_label_to_key[v.get()]
            elif k == "Inflation Proportional to Stacking":
                preset_dict[k] = bool(v.get())
            else:
                preset_dict[k] = v.get()
    presets[name] = preset_dict
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)
     # Run full pipeline (SVG output) with current image and parameters
    params = {k: v.get() for k, v in param_vars.items()}
    stack_order_dict = {label: key for label, key in STACK_ORDER_OPTIONS}
    chosen_stack_order = stack_order_dict[stack_order_var.get()]
    create_directional_bridged_svg(
        params["Image Path"],
        output_path=param_vars["Output SVG"].get(),
        n_segments=int(param_vars["Segments"].get()),
        compactness=float(param_vars["Compactness"].get()),
        max_colors=int(param_vars["Max Colors"].get()),
        bridge_distance=float(param_vars["Bridge Distance"].get()),
        color_tolerance=float(param_vars["Color Tolerance"].get()),
        proximity_threshold=float(param_vars["Proximity Threshold"].get()),
        falloff_radius=int(param_vars["Falloff Radius"].get()),
        max_curvature=np.radians(float(param_vars["Max Curvature"].get())),
        smooth_iterations=int(param_vars["Smooth Iterations"].get()),
        smooth_alpha=float(param_vars["Smooth Alpha"].get()),
        shape_order=chosen_stack_order,
        enable_grid_merge=param_vars["Enable Grid-like Blob Merging"].get(),
        area_threshold=float(param_vars["Grid Area Threshold"].get()),
        compactness_threshold=float(param_vars["Grid Compactness Threshold"].get()),
        inflation_amount=float(param_vars["Blob Inflation Amount"].get()),
        far_point_factor=float(param_vars["Far Point Inflation Factor"].get()),
        inflation_stacking=param_vars["Inflation Proportional to Stacking"].get()
    )
    # Generate thumbnail from SVG
    svg_path = params["Output SVG"]
    thumb_path = os.path.join(THUMBS_DIR, f"{name}.png")
    svg_to_thumbnail(svg_path, thumb_path)    
    update_presets_menu()
    status_var.set(f"Preset '{name}' saved.")

def load_preset(name):
    if name not in presets:
        status_var.set(f"Preset '{name}' not found.")
        return
    for k, v in presets[name].items():
        if k in param_vars and k in PRESET_PARAMS:
            # For stacking order, map from key to label for OptionMenu display
            if k == "Stacking Order":
                if v in stack_order_key_to_label:
                    param_vars[k].set(stack_order_key_to_label[v])
                else:
                    param_vars[k].set(v)  # fallback if already a label
            elif k == "Inflation Proportional to Stacking":
                param_vars[k].set(bool(v))
            else:
                param_vars[k].set(v)
    status_var.set(f"Preset '{name}' loaded.")
    show_image_preview(param_vars["Image Path"].get())

def update_preset():
    name = preset_menu_var.get()
    if not name or name not in presets:
        status_var.set("Select a preset to update.")
        return
    preset_dict = {}
    for k, v in param_vars.items():
        if k in PRESET_PARAMS:
            if k == "Stacking Order":
                preset_dict[k] = stack_order_label_to_key[v.get()]
            else:
                preset_dict[k] = v.get()
    presets[name] = preset_dict
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)
    status_var.set(f"Preset '{name}' updated.")

def delete_preset():
    name = preset_menu_var.get()
    if not name or name not in presets:
        status_var.set("Select a preset to delete.")
        return
    del presets[name]
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)
    update_presets_menu()
    status_var.set(f"Preset '{name}' deleted.")

def update_presets_menu():
    menu = preset_menu['menu']
    menu.delete(0, 'end')
    for name in presets:
        menu.add_command(label=name, command=lambda v=name: [preset_menu_var.set(v), load_preset(v)])

def svg_to_thumbnail(svg_path, thumb_path, size=(128, 128)):
    # Convert SVG to PNG (using cairosvg)
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=size[0], output_height=size[1])
    # Load into PIL and save thumbnail
    thumb_img = Image.open(io.BytesIO(png_bytes))
    thumb_img.thumbnail(size, Image.LANCZOS)
    thumb_img.save(thumb_path)

themed_label(param_frame, "Preset Name").grid(row=row, column=0, sticky="e")
themed_entry(param_frame, preset_name_var).grid(row=row, column=1)
themed_button(param_frame, "Save Preset", save_preset).grid(row=row, column=2)
row += 1

preset_menu_var = tk.StringVar()
preset_menu = tk.OptionMenu(param_frame, preset_menu_var, *presets.keys(), command=load_preset)
preset_menu.config(bg=theme["button_bg"], fg=theme["button_fg"], font=default_font)
preset_menu.grid(row=row, column=0)
themed_button(param_frame, "Update Preset", update_preset).grid(row=row, column=1)
themed_button(param_frame, "Delete Preset", delete_preset).grid(row=row, column=2)
row += 1

update_presets_menu()

# --- Previews ---
themed_label(preview_frame, "Input Preview:").pack(anchor="w")
img_preview_label = tk.Label(preview_frame, bg=theme["background"])
img_preview_label.pack(pady=2)
themed_label(preview_frame, "SVG Preview:").pack(anchor="w", pady=(10, 0))
svg_preview_label = tk.Label(preview_frame, bg=theme["background"])
svg_preview_label.pack(pady=2)

def select_image():
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if filename:
        param_vars["Image Path"].set(filename)
        show_image_preview(filename)

def select_output():
    filename = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files", "*.svg")])
    if filename:
        param_vars["Output SVG"].set(filename)

def show_image_preview(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((300, 300))
        preview_img = ImageTk.PhotoImage(img)
        img_preview_label.config(image=preview_img)
        img_preview_label.image = preview_img
    except Exception:
        img_preview_label.config(image=None)
        img_preview_label.image = None

def show_svg_preview(svg_path):
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_string = f.read()
        png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), output_width=300, output_height=300)
        img = Image.open(io.BytesIO(png_bytes))
        preview_img = ImageTk.PhotoImage(img)
        svg_preview_label.config(image=preview_img)
        svg_preview_label.image = preview_img
    except Exception:
        svg_preview_label.config(image=None)
        svg_preview_label.image = None


# --- Progress bar ---
progress = ttk.Progressbar(param_frame, mode='determinate', maximum=100)
progress.grid(row=row, column=0, columnspan=3, sticky='we', pady=5)
row += 1

# --- Run and Status ---
def int_or_none(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None

def run_vectoriser():
    def do_run():
        progress['value'] = 0
        status_var.set("Generation started...")
        progress.update_idletasks()
        # Map user-friendly label to function argument
        stack_order_dict = {label: key for label, key in STACK_ORDER_OPTIONS}
        chosen_stack_order = stack_order_dict[stack_order_var.get()]
        time.sleep(0.2)  # Optional: a brief pause for user feedback
        try:
            image = load_image(param_vars["Image Path"].get(), resize_width=int_or_none(param_vars["Resize Width"].get()))
            def progress_callback(value):
                progress['value'] = value
                progress.update_idletasks()
            create_directional_bridged_svg(
                image,
                output_path=param_vars["Output SVG"].get(),
                n_segments=int(param_vars["Segments"].get()),
                compactness=float(param_vars["Compactness"].get()),
                max_colors=int(param_vars["Max Colors"].get()),
                bridge_distance=float(param_vars["Bridge Distance"].get()),
                color_tolerance=float(param_vars["Color Tolerance"].get()),
                proximity_threshold=float(param_vars["Proximity Threshold"].get()),
                falloff_radius=int(param_vars["Falloff Radius"].get()),
                max_curvature=np.radians(float(param_vars["Max Curvature"].get())),
                smooth_iterations=int(param_vars["Smooth Iterations"].get()),
                smooth_alpha=float(param_vars["Smooth Alpha"].get()),
                progress_callback=progress_callback,
                shape_order=chosen_stack_order,
                enable_grid_merge=param_vars["Enable Grid-like Blob Merging"].get(),
                area_threshold=float(param_vars["Grid Area Threshold"].get()),
                compactness_threshold=float(param_vars["Grid Compactness Threshold"].get()),
                inflation_amount=float(param_vars["Blob Inflation Amount"].get()),
                far_point_factor=float(param_vars["Far Point Inflation Factor"].get()),
                inflation_stacking=param_vars["Inflation Proportional to Stacking"].get()
            )
            show_svg_preview(param_vars["Output SVG"].get())
            status_var.set(f"SVG generated: {param_vars['Output SVG'].get()}")
            progress['value'] = 100
            progress.update_idletasks()
            save_to_history()
        except Exception as e:
            progress['value'] = 0
            progress.update_idletasks()
            status_var.set(f"Error: {str(e)}")
    threading.Thread(target=do_run).start()

    def save_to_history():
        global parameter_history, history_index
        # Capture all parameter values (except paths if you wish)
        current_params = {k: v.get() for k, v in param_vars.items()}
        # If you undid and then generate new, cut off any "future" history
        if history_index < len(parameter_history) - 1:
            parameter_history = parameter_history[:history_index+1]
        parameter_history.append(current_params)
        history_index = len(parameter_history) - 1

def restore_from_history(idx):
    for k, v in parameter_history[idx].items():
        if k in param_vars:
            param_vars[k].set(v)

row_stack = row  # e.g. insert just before Run button
themed_label(param_frame, "Stacking Order").grid(row=row_stack, column=0, sticky="e", pady=2)
stack_order_menu = tk.OptionMenu(
    param_frame, stack_order_var,
    *[label for label, key in STACK_ORDER_OPTIONS]
)
stack_order_menu.config(bg=theme["button_bg"], fg=theme["button_fg"], font=default_font)
stack_order_menu.grid(row=row_stack, column=1, pady=2)
row +=1

themed_button(param_frame, "Run", run_vectoriser).grid(row=row, column=1, pady=10)
tk.Label(param_frame, textvariable=status_var, fg=theme["label_fg"], bg=theme["background"], font=default_font).grid(row=row+1, column=0, columnspan=3, sticky='w')

def undo_history():
    global history_index
    if history_index > 0:
        history_index -= 1
        restore_from_history(history_index)
        status_var.set(f"History: step {history_index+1}/{len(parameter_history)}")
    else:
        status_var.set("No earlier history.")

def redo_history():
    global history_index
    if history_index < len(parameter_history) - 1:
        history_index += 1
        restore_from_history(history_index)
        status_var.set(f"History: step {history_index+1}/{len(parameter_history)}")
    else:
        status_var.set("No later history.")

themed_button(root, "Undo", undo_history).grid(row=0, column=3)
themed_button(root, "Redo", redo_history).grid(row=0, column=4)
root.bind('<Control-z>', lambda e: undo_history())
root.bind('<Control-y>', lambda e: redo_history())
status_var.set(f"History: step {history_index+1}/{len(parameter_history)}")

root.mainloop()
