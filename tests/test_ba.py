import numpy as np
from scipy.spatial.transform import Rotation

from organ_slam.ba import bundle_adjust
from organ_slam.geometry import make_T, project


def _toy(seed=0, n_cams=5, n_pts=60):
    rng = np.random.default_rng(seed)
    K = np.array([[200.0, 0, 160], [0, 200.0, 160], [0, 0, 1]])
    X = rng.uniform([-2, -2, 6], [2, 2, 10], (n_pts, 3))
    poses = np.array([make_T(Rotation.from_rotvec(rng.normal(0, 0.05, 3)).as_matrix(),
                             [0.4 * i, rng.normal(0, 0.1), rng.normal(0, 0.1)]) for i in range(n_cams)])
    cam_idx = np.repeat(np.arange(n_cams), n_pts)
    pt_idx = np.tile(np.arange(n_pts), n_cams)
    obs = np.vstack([project(K, T, X)[0] for T in poses])
    return K, poses, X, cam_idx, pt_idx, obs, rng


def test_ba_recovers_perturbed_problem():
    K, poses, X, cam_idx, pt_idx, obs, rng = _toy()
    noisy_poses = poses.copy()
    for i in range(2, len(poses)):
        noisy_poses[i] = make_T(Rotation.from_rotvec(rng.normal(0, 0.02, 3)).as_matrix(), rng.normal(0, 0.05, 3)) @ poses[i]
    noisy_X = X + rng.normal(0, 0.1, X.shape)
    fixed = np.zeros(len(poses), bool)
    fixed[:2] = True
    out_poses, out_X, err = bundle_adjust(K, noisy_poses, noisy_X, cam_idx, pt_idx, obs, fixed, max_iters=30)
    assert err.max() < 1e-3
    assert np.abs(out_X - X).max() < 1e-3
    assert np.abs(out_poses - poses).max() < 1e-3


def test_ba_rejects_outlier_observation():
    K, poses, X, cam_idx, pt_idx, obs, rng = _toy(2)
    obs = obs + rng.normal(0, 0.3, obs.shape)
    obs[7] += [40.0, -25.0]
    fixed = np.zeros(len(poses), bool)
    fixed[:2] = True
    out_poses, out_X, err = bundle_adjust(K, poses, X + rng.normal(0, 0.05, X.shape), cam_idx, pt_idx, obs, fixed)
    assert err[7] > 20                       # the outlier is not fitted
    assert np.median(err) < 0.6
    others = np.arange(len(X)) != pt_idx[7]
    assert np.abs(out_X - X)[others].max() < 0.5      # depth noise from 0.3 px observation noise
