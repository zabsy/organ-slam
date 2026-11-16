"""Trajectory and map accuracy against ground truth.

Monocular SLAM has no absolute scale, so the estimate is first aligned to the
ground truth with a similarity transform (scale + rotation + translation).
"""
import numpy as np

from .geometry import umeyama


def evaluate_trajectory(est_wc, gt_wc):
    """est_wc: {frame_id: T_wc}. gt_wc: (N,4,4). Returns a metrics dict (+ aligned centres)."""
    ids = np.array(sorted(est_wc))
    est = np.array([est_wc[i][:3, 3] for i in ids])
    gt = gt_wc[ids, :3, 3]
    s, R, t = umeyama(est, gt)
    aligned = s * est @ R.T + t
    err = np.linalg.norm(aligned - gt, axis=1)
    path = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    # Rotation error after alignment.
    rot = []
    for i in ids:
        dR = gt_wc[i][:3, :3].T @ (R @ est_wc[i][:3, :3])
        rot.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    # Scale drift: similarity scale fitted on the first vs last third of the run.
    k = max(len(ids) // 3, 3)
    s_first = umeyama(est[:k], gt[:k])[0]
    s_last = umeyama(est[-k:], gt[-k:])[0]
    return dict(frame_ids=ids, aligned=aligned, gt=gt, err=err, sim3=(s, R, t),
                ate_rmse=float(np.sqrt(np.mean(err ** 2))), ate_median=float(np.median(err)),
                ate_max=float(err.max()), path_length=path,
                ate_pct=float(100 * np.sqrt(np.mean(err ** 2)) / path),
                rot_err_mean_deg=float(np.mean(rot)), scale_drift=float(s_last / s_first),
                tracked=len(ids), total=len(gt_wc))


def evaluate_map(points, sim3, scene):
    """Distance of the aligned sparse map points to the true synthetic wall."""
    s, R, t = sim3
    aligned = s * points @ R.T + t
    d = np.abs(scene.sdf(aligned)[0])
    return dict(aligned=aligned, n_points=len(points), map_median=float(np.median(d)),
                map_p90=float(np.percentile(d, 90)))
