#!/usr/bin/env python3
"""
Visualize a .npy point cloud with labels using Open3D.

Expected .npy format per point: [x, y, z, r, g, b, label]
 - xyz: float32/float64 in world units
 - rgb: uint8 (0-255) or float (0-1)
 - label: int (or float castable to int)

Usage examples:
  python tools/view_npy_pointcloud.py data/bim/Area_1_conferenceRoom_1_s.npy
  python tools/view_npy_pointcloud.py data/bim/sample.npy --color rgb --voxel-size 0.01
  python tools/view_npy_pointcloud.py data/bim/sample.npy --screenshot outputs/preview.png
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception as open3d_import_error:  # pragma: no cover
    raise SystemExit(
        "Failed to import open3d. Please install it: pip install open3d"
    ) from open3d_import_error

try:
    import matplotlib
    import matplotlib.cm as cm
except Exception:
    matplotlib = None
    cm = None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize .npy point cloud with labels (xyzrgb label) using Open3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default="data/bim/Area_4_storage_1_S.npy", help="Path to .npy file (shape: N x 7)")
    parser.add_argument(
        "--color",
        choices=["label", "rgb"],
        default="label",
        help="Color points by label or by stored RGB values",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Optional voxel downsample size (in same units as xyz)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional random downsampling to at most this many points",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="black",
        help="Background color: 'black', 'white', or hex like #202020",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Point size in the visualizer",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Optional path to save a screenshot (PNG)",
    )
    return parser.parse_args()


def load_npy_point_cloud(npy_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"File not found: {npy_path}")
    data = np.load(npy_path)
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(
            f"Expected shape (N, 7+) for xyzrgb+label, got {data.shape}"
        )
    xyz = data[:, 0:3].astype(np.float64)
    rgb = data[:, 3:6]
    labels = data[:, 6]
    # Normalize RGB to 0-1
    if np.issubdtype(rgb.dtype, np.integer):
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)
        # If values look like 0-255 floats, normalize
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
    # Ensure labels are int
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)
    return xyz, rgb, labels


def generate_label_colors(labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, Tuple[float, float, float]]]:
    unique_labels = np.unique(labels)
    # Prefer matplotlib tab20 for distinct colors
    if cm is not None:
        colormap = cm.get_cmap("tab20", max(20, unique_labels.size))
        color_table = {
            int(lbl): tuple(colormap(i % colormap.N)[:3]) for i, lbl in enumerate(unique_labels)
        }
    else:
        # Fallback deterministic hashing to colors
        rng = np.random.default_rng(42)
        color_table = {int(lbl): tuple(rng.random(3)) for lbl in unique_labels}

    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
    for lbl, color in color_table.items():
        colors[labels == lbl] = color
    return colors, color_table


def color_string_to_rgb(background: str) -> Tuple[float, float, float]:
    name = background.strip().lower()
    if name == "black":
        return (0.0, 0.0, 0.0)
    if name == "white":
        return (1.0, 1.0, 1.0)
    if name.startswith("#") and len(name) in (4, 7):
        if len(name) == 4:
            r = int(name[1] * 2, 16)
            g = int(name[2] * 2, 16)
            b = int(name[3] * 2, 16)
        else:
            r = int(name[1:3], 16)
            g = int(name[3:5], 16)
            b = int(name[5:7], 16)
        return (r / 255.0, g / 255.0, b / 255.0)
    return (0.0, 0.0, 0.0)


def build_point_cloud(
    xyz: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def maybe_random_downsample(
    xyz: np.ndarray, colors: np.ndarray, labels: np.ndarray, max_points: int | None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_points is None or xyz.shape[0] <= max_points:
        return xyz, colors, labels
    rng = np.random.default_rng(0)
    indices = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[indices], colors[indices], labels[indices]


def visualize(
    cloud: o3d.geometry.PointCloud,
    background_rgb: Tuple[float, float, float],
    point_size: float,
    screenshot_path: str | None,
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cloud)

    render_opts = vis.get_render_option()
    render_opts.point_size = max(1.0, float(point_size))
    render_opts.background_color = np.asarray(background_rgb, dtype=np.float32)

    vis.get_view_control().set_zoom(0.8)

    if screenshot_path:
        # Render one frame, capture, then keep window open for interaction
        vis.poll_events()
        vis.update_renderer()
        os.makedirs(os.path.dirname(screenshot_path) or ".", exist_ok=True)
        vis.capture_screen_image(screenshot_path, do_render=True)
        print(f"Saved screenshot to: {screenshot_path}")

    vis.run()
    vis.destroy_window()


def main() -> None:
    args = parse_arguments()

    xyz, rgb, labels = load_npy_point_cloud(args.input)
    num_points = xyz.shape[0]
    print(f"Loaded {num_points} points from: {args.input}")

    if args.color == "label":
        colors, color_table = generate_label_colors(labels)
        # Show mapping in console for reference
        sorted_items = sorted(color_table.items(), key=lambda kv: kv[0])
        print("Label -> Color mapping (RGB in 0-1):")
        for lbl, (r, g, b) in sorted_items:
            print(f"  {lbl:>4}: ({r:.3f}, {g:.3f}, {b:.3f})")
    else:
        colors = rgb

    # Optional random downsample before building cloud for speed
    xyz, colors, labels = maybe_random_downsample(xyz, colors, labels, args.max_points)

    cloud = build_point_cloud(xyz, colors)

    # Optional voxel downsample using Open3D
    if args.voxel_size is not None and args.voxel_size > 0:
        print(f"Applying voxel downsampling with voxel_size={args.voxel_size}")
        cloud = cloud.voxel_down_sample(voxel_size=float(args.voxel_size))

    bg = color_string_to_rgb(args.background)
    visualize(cloud, bg, args.point_size, args.screenshot)


if __name__ == "__main__":
    main()


