"""
coverage_map.py — 2D grid tracking which cells have been explored by the swarm.

Each cell represents a (cell_size × cell_size) metre square on the XY plane.
A cell is marked visited when any drone passes within sensor_radius of its centre.
"""

import numpy as np


class CoverageMap:
    def __init__(self, grid_size: int = 10, cell_size: float = 1.0, sensor_radius: float = 1.5):
        """
        Args:
            grid_size:     Number of cells per side (grid is grid_size × grid_size).
            cell_size:     Physical size of each cell in metres.
            sensor_radius: A cell is marked visited when a drone is within this radius (m).
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.sensor_radius = sensor_radius

        # Physical extent: [0, grid_size * cell_size] in x and y
        self.arena_size = grid_size * cell_size

        self._grid = np.zeros((grid_size, grid_size), dtype=np.float32)  # 0=unexplored, 1=visited
        self._total_cells = grid_size * grid_size

        # Pre-compute cell centres for vectorised distance checks
        xs = (np.arange(grid_size) + 0.5) * cell_size
        ys = (np.arange(grid_size) + 0.5) * cell_size
        cx, cy = np.meshgrid(xs, ys, indexing="ij")  # shape (grid_size, grid_size)
        self._cell_centres = np.stack([cx, cy], axis=-1)  # (G, G, 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the grid — call at the start of each episode."""
        self._grid[:] = 0.0

    def update(self, positions: np.ndarray) -> float:
        """
        Mark cells within sensor_radius of any drone as visited.

        Args:
            positions: (N, 2) or (N, 3) array of drone XY[Z] world positions.

        Returns:
            delta: fraction of NEW cells marked this step (0 to 1).
        """
        positions_xy = np.asarray(positions, dtype=np.float32)
        if positions_xy.ndim == 1:
            positions_xy = positions_xy[np.newaxis]
        positions_xy = positions_xy[:, :2]  # only x, y

        prev_visited = self._grid.sum()

        # For each drone, find cells within sensor_radius
        for pos in positions_xy:
            dist = np.linalg.norm(self._cell_centres - pos, axis=-1)  # (G, G)
            self._grid[dist <= self.sensor_radius] = 1.0

        new_visited = self._grid.sum()
        delta = float(new_visited - prev_visited) / self._total_cells
        return delta

    def coverage_pct(self) -> float:
        """Return fraction of cells visited [0.0, 1.0]."""
        return float(self._grid.sum()) / self._total_cells

    def local_coverage_pct(self, position: np.ndarray, window: int = 10) -> float:
        """
        Return fraction of cells visited within a (window × window) cell window
        centred on the given drone position. Used as part of per-drone observation.
        """
        pos_xy = np.asarray(position[:2], dtype=np.float32)
        # Convert world position to grid indices
        ci = int(np.clip(pos_xy[0] / self.cell_size, 0, self.grid_size - 1))
        cj = int(np.clip(pos_xy[1] / self.cell_size, 0, self.grid_size - 1))

        half = window // 2
        i0, i1 = max(0, ci - half), min(self.grid_size, ci + half)
        j0, j1 = max(0, cj - half), min(self.grid_size, cj + half)

        patch = self._grid[i0:i1, j0:j1]
        if patch.size == 0:
            return 0.0
        return float(patch.mean())

    def nearest_uncovered_direction(self, position: np.ndarray) -> np.ndarray:
        """
        Return a unit vector (x, y) pointing toward the nearest uncovered cell.
        Returns zero vector if all cells are covered.
        """
        pos_xy = np.asarray(position[:2], dtype=np.float32)
        uncovered_mask = self._grid == 0.0
        if not uncovered_mask.any():
            return np.zeros(2, dtype=np.float32)

        uncovered_centres = self._cell_centres[uncovered_mask]  # (M, 2)
        diffs = uncovered_centres - pos_xy                       # (M, 2)
        dists = np.linalg.norm(diffs, axis=-1)                   # (M,)
        nearest_diff = diffs[np.argmin(dists)]

        norm = np.linalg.norm(nearest_diff)
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (nearest_diff / norm).astype(np.float32)

    def get_grid(self) -> np.ndarray:
        """Return a copy of the current grid (grid_size × grid_size float32)."""
        return self._grid.copy()

    def get_flat_grid(self) -> np.ndarray:
        """Return flattened grid — used in global state for centralized critic."""
        return self._grid.flatten()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _world_to_grid(self, pos_xy: np.ndarray):
        """Convert (x, y) world coords to (col, row) grid indices (clamped)."""
        col = int(np.clip(pos_xy[0] / self.cell_size, 0, self.grid_size - 1))
        row = int(np.clip(pos_xy[1] / self.cell_size, 0, self.grid_size - 1))
        return col, row
