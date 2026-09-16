"""Privileged quantitative metrics for planar swarm tasks."""

from __future__ import annotations

import numpy as np


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)


def largest_component_fraction(
    positions: np.ndarray, communication_range_m: float
) -> float:
    """Fraction of robots in the largest undirected proximity component."""

    if communication_range_m <= 0.0:
        raise ValueError("communication_range_m must be positive")
    count = len(positions)
    if count == 0:
        return 0.0
    distances = pairwise_distances(positions)
    adjacency = (distances <= communication_range_m) & (distances > 0.0)
    unseen = set(range(count))
    largest = 0
    while unseen:
        seed = unseen.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for neighbor in np.flatnonzero(adjacency[current]):
                neighbor_id = int(neighbor)
                if neighbor_id in unseen:
                    unseen.remove(neighbor_id)
                    component.add(neighbor_id)
                    frontier.append(neighbor_id)
        largest = max(largest, len(component))
    return largest / count


def swarm_radius_m(positions: np.ndarray) -> float:
    """Root-mean-square distance from the swarm centroid."""

    points = np.asarray(positions, dtype=np.float64)
    if len(points) == 0:
        return 0.0
    centroid = points.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))


def polarization(headings_rad: np.ndarray) -> float:
    """Magnitude of the mean unit heading, in [0, 1]."""

    headings = np.asarray(headings_rad, dtype=np.float64)
    if headings.size == 0:
        return 0.0
    vectors = np.column_stack((np.cos(headings), np.sin(headings)))
    return float(np.linalg.norm(vectors.mean(axis=0)))


def grid_coverage_fraction(
    positions: np.ndarray,
    arena_size_m: tuple[float, float],
    footprint_radius_m: float,
    grid_shape: tuple[int, int] = (18, 12),
) -> float:
    """Fraction of evaluation cells within a robot's sensing footprint."""

    if footprint_radius_m <= 0.0:
        raise ValueError("footprint_radius_m must be positive")
    width, height = arena_size_m
    columns, rows = grid_shape
    xs = np.linspace(-width / 2.0, width / 2.0, columns)
    ys = np.linspace(-height / 2.0, height / 2.0, rows)
    cells = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    distances = np.linalg.norm(cells[:, None, :] - positions[None, :, :], axis=-1)
    return float(np.mean(np.min(distances, axis=1) <= footprint_radius_m))


def mean_nearest_neighbor_distance_m(positions: np.ndarray) -> float:
    points = np.asarray(positions, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    distances = pairwise_distances(points)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances, axis=1).mean())
