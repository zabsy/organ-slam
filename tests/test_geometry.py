import numpy as np
from scipy.spatial.transform import Rotation

from organ_slam.geometry import (epipolar_dist, fundamental_from_poses, inv_T, make_T, parallax_deg,
                           project, triangulate, umeyama)

K = np.array([[200.0, 0, 160], [0, 200.0, 160], [0, 0, 1]])


def _two_views():
    rng = np.random.default_rng(0)
    X = rng.uniform([-2, -2, 5], [2, 2, 9], (40, 3))
    T1 = np.eye(4)
    T2 = make_T(Rotation.from_rotvec([0.05, -0.1, 0.02]).as_matrix(), [-1.0, 0.1, 0.2])
    return X, T1, T2


def test_inverse_pose():
    _, _, T2 = _two_views()
    assert np.allclose(inv_T(T2) @ T2, np.eye(4))


def test_triangulation_roundtrip():
    X, T1, T2 = _two_views()
    x1, x2 = project(K, T1, X)[0], project(K, T2, X)[0]
    assert np.allclose(triangulate(K, T1, T2, x1, x2), X, atol=1e-6)
    assert np.all(parallax_deg(T1, T2, X) > 1.0)


def test_epipolar_constraint():
    X, T1, T2 = _two_views()
    x1, x2 = project(K, T1, X)[0], project(K, T2, X)[0]
    F = fundamental_from_poses(K, T1, T2)
    assert epipolar_dist(F, x1, x2).max() < 1e-6
    assert epipolar_dist(F, x1, x2 + [0, 5.0]).mean() > 1.0


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3))
    R = Rotation.from_rotvec([0.3, -0.5, 0.8]).as_matrix()
    dst = 2.5 * src @ R.T + [1.0, -2.0, 3.0]
    s, R_est, t = umeyama(src, dst)
    assert np.isclose(s, 2.5) and np.allclose(R_est, R) and np.allclose(t, [1.0, -2.0, 3.0])
