import numpy as np


class ScanSimConfig:
    def __init__(
        self,
        h_fov_min_deg: float = -180.0,
        h_fov_max_deg: float = 180.0,
        v_fov_min_deg: float = -25.0,
        v_fov_max_deg: float = 2.0,
        h_res_deg: float = 0.35,
        v_res_deg: float = 0.4,
        max_range: float = 30.0,
        range_noise_std: float = 0.01,
        dropout_prob: float = 0.05,
        sensor_height: float = 1.2,
        sensor_offset_xy: tuple = (0.0, -0.5),
    ) -> None:
        self.h_fov_min_deg = h_fov_min_deg
        self.h_fov_max_deg = h_fov_max_deg
        self.v_fov_min_deg = v_fov_min_deg
        self.v_fov_max_deg = v_fov_max_deg
        self.h_res_deg = max(1e-6, h_res_deg)
        self.v_res_deg = max(1e-6, v_res_deg)
        self.max_range = max_range
        self.range_noise_std = range_noise_std
        self.dropout_prob = np.clip(dropout_prob, 0.0, 1.0)
        self.sensor_height = sensor_height
        self.sensor_offset_xy = sensor_offset_xy


def _cartesian_to_spherical(points_xyz: np.ndarray) -> tuple:
    # Returns r, azimuth_deg, elevation_deg
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    r = np.linalg.norm(points_xyz, axis=1) + 1e-9
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))
    return r, az, el


def _spherical_to_cartesian(r: np.ndarray, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.stack([x, y, z], axis=1)


def simulate_scan_block(
    block_points_xyz_abs: np.ndarray,
    block_center_abs: np.ndarray,
    config: ScanSimConfig,
) -> tuple:
    """
    Approximate a LiDAR scan over a block of points by:
    1) Placing a sensor near the block center
    2) Applying FOV and range masking
    3) Angular downsampling by keeping the closest point per (az, el) bin
    4) Adding small range noise and random dropout

    Inputs are in absolute coordinates (meters).
    Returns (filtered points in absolute coordinates, kept_indices w.r.t. input block array).
    """
    if block_points_xyz_abs.size == 0:
        return block_points_xyz_abs, np.arange(block_points_xyz_abs.shape[0], dtype=np.int64)

    # Sensor pose relative to block center
    sensor_xy = np.array([
        block_center_abs[0] + config.sensor_offset_xy[0],
        block_center_abs[1] + config.sensor_offset_xy[1],
    ])
    sensor_pos = np.array([sensor_xy[0], sensor_xy[1], block_center_abs[2] + config.sensor_height], dtype=np.float32)

    # Translate points to sensor frame
    pts_sensor = block_points_xyz_abs.astype(np.float32) - sensor_pos[None, :]

    # Spherical coordinates
    r, az_deg, el_deg = _cartesian_to_spherical(pts_sensor)

    # FOV and range masking
    h_ok = (az_deg >= config.h_fov_min_deg) & (az_deg <= config.h_fov_max_deg)
    v_ok = (el_deg >= config.v_fov_min_deg) & (el_deg <= config.v_fov_max_deg)
    d_ok = r <= config.max_range
    mask = h_ok & v_ok & d_ok
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    r = r[mask]
    az_deg = az_deg[mask]
    el_deg = el_deg[mask]
    pts_sensor = pts_sensor[mask]
    idxs = np.nonzero(mask)[0]

    # Angular binning (keep nearest per bin)
    h_bins = int(np.ceil((config.h_fov_max_deg - config.h_fov_min_deg) / config.h_res_deg))
    v_bins = int(np.ceil((config.v_fov_max_deg - config.v_fov_min_deg) / config.v_res_deg))
    h_bin = np.floor((az_deg - config.h_fov_min_deg) / config.h_res_deg).astype(np.int32)
    v_bin = np.floor((el_deg - config.v_fov_min_deg) / config.v_res_deg).astype(np.int32)
    h_bin = np.clip(h_bin, 0, max(0, h_bins - 1))
    v_bin = np.clip(v_bin, 0, max(0, v_bins - 1))
    lin_bin = v_bin * max(1, h_bins) + h_bin

    order = np.argsort(r)
    keep_mask_local = np.zeros(order.shape[0], dtype=bool)
    seen = set()
    for pos in order:
        lb = lin_bin[pos]
        if lb in seen:
            continue
        seen.add(lb)
        keep_mask_local[pos] = True

    r = r[keep_mask_local]
    az_deg = az_deg[keep_mask_local]
    el_deg = el_deg[keep_mask_local]
    kept_block_indices = idxs[keep_mask_local]

    # Add range noise
    if config.range_noise_std > 0:
        r = r + np.random.normal(0.0, config.range_noise_std, size=r.shape[0]).astype(np.float32)
        r = np.clip(r, 0.0, config.max_range)

    # Random dropout
    if config.dropout_prob > 0:
        keep = np.random.rand(r.shape[0]) > config.dropout_prob
        if np.any(keep):
            r = r[keep]
            az_deg = az_deg[keep]
            el_deg = el_deg[keep]
            kept_block_indices = kept_block_indices[keep]
        else:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    # Back to sensor frame cartesian and then to absolute
    pts_sensor_sim = _spherical_to_cartesian(r, az_deg, el_deg).astype(np.float32)
    pts_abs_sim = pts_sensor_sim + sensor_pos[None, :]
    return pts_abs_sim, kept_block_indices


