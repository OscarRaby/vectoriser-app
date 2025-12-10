Porting plan: Python Tkinter organic segmentation vectoriser -> C# (WPF)

Goal
- Feature parity with organic_segmentation_gui.py: UI controls, background pipeline, SVG export/preview, presets/history, droplet modes.

Stack choices
- UI: WPF on .NET 8 (desktop-only).
- Imaging/numerics: OpenCvSharp for image ops + contours; MathNet.Numerics for linear algebra; LibNoise.Net for Perlin; optional Accord.NET (KMeans) if ML.NET overkill.
- SVG: Svg.Skia or Svg.NET; SkiaSharp for raster preview; System.Text.Json for presets.

Architecture
- MVVM: ViewModels for parameters/modifiers/history; Commands for Run/Preview/Undo/Redo; Background tasks with Dispatcher updates.
- Services: ImageService (load/resize), SegmentationService, ContourService, DropletService, SvgExportService, PreviewService.
- Models: ParameterSet, ModifierFlags, DropletDescriptor (polygon/ellipse/rect), ContourData (points, centroid, color), PipelineResult.

UI layout (WPF)
- 3 columns Grid: left scrollable controls (sliders/check/checklist), middle buttons/status/presets/history, right preview (SkiaSharp canvas or Image bound to rendered bitmap).
- Controls map 1:1 to Tk params (Noise Scale, Blur Sigma, Compactness, Max Colors, Bridge Distance, Color Tolerance, Proximity Threshold, Falloff Radius, Max Curvature, Smooth Iterations, Smooth Alpha, Blob Inflation Amount, Far Point Inflation Factor, Inflation Proportional, Stacking Order, Segmentation Multiplier, droplet controls, painterly toggles, SVG path, vector preview toggle).
- Status TextBlock + ProgressBar + Run/Preview buttons; presets dropdown/save/update/delete; Undo/Redo.

Pipeline parity tasks
- Image load/resize to fit preview; store float32 [0,1] arrays.
- Segmentation: noise-modulated watershed (Gaussian blur + Perlin noise). Use OpenCvSharp: convert to gray, GaussianBlur, add noise field, find local minima (distance transform or min filter), watershed with markers. Scale parameters by image size and multiplier.
- Color quantization: KMeans (Accord.NET/ML.NET) on pixels -> palette and remap.
- Contours: find contours per label (OpenCvSharp FindContours), compute centroid, majority color.
- Bridging: KD-tree (MathNet KD?) or custom k-d via Accord; connect nearby similarly colored contours; enforce curvature and falloff; apply displacements.
- Inflation: radial push from centroid with far_point_factor; proportional-to-stacking variant.
- Smoothing: Laplacian smoothing iterations over contour points.
- Droplets: Painterly polygons/ellipses/rects with spread/distance/size/std; Organic droplets toward neighbor centroids with brightness gate, jitter, elongation, strength, simplify via RDP.
- Stacking order: area, brightness, position_x/y, centre variants.
- SVG export: contours to path (quantize/simplify), droplets to polygon/ellipse/rect; set fill colors; write viewBox; fallback raw writer.
- Preview: lightweight 150x150 crop run (no droplets) and optional vector overlay; rasterize SVG for preview (Svg.Skia->SkiaSharp bitmap) or draw primitives directly.

History & presets
- Presets JSON file compatible keys; Undo/Redo stacks of ParameterSet+ModifierFlags snapshots; consider command-based history.

Threading
- Run/Preview execute on Task.Run; UI updates via Dispatcher; disable Run during processing; progress reporting hooks through IProgress.

Determinism/testing
- Seedable RNG for droplets/segmentation; unit tests for parameter scaling, contour ordering, droplet generation, SVG serialization.

Initial milestones
1) Scaffold WPF app with MVVM, parameter ViewModel, bindings, Run/Preview buttons, status/progress, image load. Placeholder services.
2) Implement image load/resize + preview display; parameter persistence (JSON); undo/redo skeleton.
3) Implement segmentation (noise-watershed) and contour extraction; verify labels/contours drawn.
4) Add quantization, bridging, inflation, smoothing; validate contour drawing order.
5) Add droplet generation (painterly + organic), stacking order, and SVG export + preview raster.
6) Polish: error handling, progress, performance, deterministic seeds, tests with sample images.
