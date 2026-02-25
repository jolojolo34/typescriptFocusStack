#!/usr/bin/env python3
"""
Interactive STL topography mapper.

Workflow:
1) Pick an STL file with a file picker dialog.
2) Enter an XY map rectangle and output grid resolution.
3) Ray-cast from +Z down to sample surface.
4) Save depth map (CSV) where rows are Y samples and columns are X samples.

Dependencies:
    pip install numpy trimesh
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


@dataclass
class MapSettings:
    center_x: float
    center_y: float
    width_x: float
    width_y: float
    points_x: int
    points_y: int


def ask_float(root: tk.Tk, title: str, prompt: str, initial: float) -> float:
    value = simpledialog.askfloat(title, prompt, parent=root, initialvalue=initial)
    if value is None:
        raise KeyboardInterrupt("User cancelled input dialog")
    return value


def ask_int(root: tk.Tk, title: str, prompt: str, initial: int, min_value: int = 1) -> int:
    value = simpledialog.askinteger(
        title,
        prompt,
        parent=root,
        initialvalue=initial,
        minvalue=min_value,
    )
    if value is None:
        raise KeyboardInterrupt("User cancelled input dialog")
    return value


def choose_stl_file(root: tk.Tk) -> Path:
    selected = filedialog.askopenfilename(
        parent=root,
        title="Select STL file",
        filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
    )
    if not selected:
        raise KeyboardInterrupt("No STL file selected")
    return Path(selected)


def choose_output_csv(root: tk.Tk, suggested_name: str) -> Path:
    selected = filedialog.asksaveasfilename(
        parent=root,
        title="Save depth map CSV",
        defaultextension=".csv",
        initialfile=suggested_name,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not selected:
        raise KeyboardInterrupt("No output CSV selected")
    return Path(selected)


def load_mesh(path: Path):
    try:
        import trimesh
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import trimesh. Install with: pip install trimesh"
        ) from exc

    mesh = trimesh.load_mesh(path, force="mesh")
    if mesh.is_empty:
        raise RuntimeError("The selected STL file appears to be empty.")

    # Normalize to a single mesh if a Scene was loaded.
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    return mesh


def ask_map_settings(root: tk.Tk, mesh) -> MapSettings:
    min_corner, max_corner = mesh.bounds
    default_cx = float((min_corner[0] + max_corner[0]) / 2)
    default_cy = float((min_corner[1] + max_corner[1]) / 2)
    default_wx = float(max_corner[0] - min_corner[0])
    default_wy = float(max_corner[1] - min_corner[1])

    center_x = ask_float(root, "Map center X", "Center X coordinate", default_cx)
    center_y = ask_float(root, "Map center Y", "Center Y coordinate", default_cy)
    width_x = ask_float(root, "Map width X", "Map width along X", default_wx)
    width_y = ask_float(root, "Map width Y", "Map width along Y", default_wy)
    points_x = ask_int(root, "Grid points X", "Number of points along X", 200, min_value=2)
    points_y = ask_int(root, "Grid points Y", "Number of points along Y", 200, min_value=2)

    if width_x <= 0 or width_y <= 0:
        raise RuntimeError("Map width values must be greater than zero.")

    return MapSettings(
        center_x=center_x,
        center_y=center_y,
        width_x=width_x,
        width_y=width_y,
        points_x=points_x,
        points_y=points_y,
    )


def compute_depth_map(mesh, settings: MapSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Build XY sample grid.
    x0 = settings.center_x - settings.width_x / 2
    x1 = settings.center_x + settings.width_x / 2
    y0 = settings.center_y - settings.width_y / 2
    y1 = settings.center_y + settings.width_y / 2

    xs = np.linspace(x0, x1, settings.points_x)
    ys = np.linspace(y0, y1, settings.points_y)
    grid_x, grid_y = np.meshgrid(xs, ys)

    sample_count = grid_x.size

    # Ray origin plane is above mesh top bound.
    z_top = float(mesh.bounds[1][2])
    z_origin = z_top + max(settings.width_x, settings.width_y, 1.0)

    origins = np.column_stack(
        [
            grid_x.reshape(sample_count),
            grid_y.reshape(sample_count),
            np.full(sample_count, z_origin, dtype=np.float64),
        ]
    )
    directions = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (sample_count, 1))

    # Avoid optional acceleration dependencies; pure-triangle intersector always works.
    from trimesh.ray.ray_triangle import RayMeshIntersector

    intersector = RayMeshIntersector(mesh)
    hit_triangles, hit_ray_ids, hit_points = intersector.intersects_id(
        ray_origins=origins,
        ray_directions=directions,
        return_locations=True,
        multiple_hits=False,
    )

    _ = hit_triangles  # kept for clarity

    hit_z = np.full(sample_count, np.nan, dtype=np.float64)
    hit_z[hit_ray_ids] = hit_points[:, 2]
    hit_z_grid = hit_z.reshape(settings.points_y, settings.points_x)

    valid = np.isfinite(hit_z_grid)
    if not np.any(valid):
        raise RuntimeError(
            "No surface intersections found for this XY area. Check map location/size."
        )

    map_top_z = float(np.nanmax(hit_z_grid))
    depth_grid = map_top_z - hit_z_grid

    return xs, ys, hit_z_grid, depth_grid


def save_csv(path: Path, xs: np.ndarray, ys: np.ndarray, z_grid: np.ndarray, depth_grid: np.ndarray) -> None:
    # CSV format: y, x, z_surface, depth_from_local_top
    x_flat = np.tile(xs, ys.size)
    y_flat = np.repeat(ys, xs.size)

    out = np.column_stack(
        [
            x_flat,
            y_flat,
            z_grid.reshape(-1),
            depth_grid.reshape(-1),
        ]
    )

    header = "x,y,z_surface,depth_from_local_top"
    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt="%.10g")


def main() -> int:
    root = tk.Tk()
    root.withdraw()

    try:
        stl_path = choose_stl_file(root)
        mesh = load_mesh(stl_path)
        settings = ask_map_settings(root, mesh)
        xs, ys, z_grid, depth_grid = compute_depth_map(mesh, settings)

        default_name = f"{stl_path.stem}_depth_map.csv"
        output_path = choose_output_csv(root, default_name)
        save_csv(output_path, xs, ys, z_grid, depth_grid)

        messagebox.showinfo(
            title="Done",
            message=(
                f"Depth map saved to:\n{output_path}\n\n"
                f"Grid: {settings.points_x} x {settings.points_y} points"
            ),
            parent=root,
        )
        return 0

    except KeyboardInterrupt:
        messagebox.showwarning("Cancelled", "Operation cancelled by user.", parent=root)
        return 1
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("Error", str(exc), parent=root)
        return 2
    finally:
        root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
