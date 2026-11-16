"""Small SE(3) / multi-view helpers. Poses are 4x4; T_cw maps world -> camera."""
import cv2
import numpy as np


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def project(K, T_cw, X):
    """Project Nx3 world points. Returns (Nx2 pixels, N depths)."""
    Xc = X @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    uv = np.stack([K[0, 0] * Xc[:, 0] / zs + K[0, 2],
                   K[1, 1] * Xc[:, 1] / zs + K[1, 2]], axis=1)
    return uv, z


def triangulate(K, T1_cw, T2_cw, x1, x2):
    """Linear triangulation of Nx2 pixel matches. Returns Nx3 world points."""
    P1 = K @ T1_cw[:3]
    P2 = K @ T2_cw[:3]
    Xh = cv2.triangulatePoints(P1, P2, x1.T.astype(np.float64), x2.T.astype(np.float64))
    return (Xh[:3] / Xh[3]).T


def parallax_deg(T1_cw, T2_cw, X):
    """Angle (degrees) subtended at each point by the two camera centres."""
    c1 = inv_T(T1_cw)[:3, 3]
    c2 = inv_T(T2_cw)[:3, 3]
    a = c1 - X
    b = c2 - X
    cosang = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def fundamental_from_poses(K, T1_cw, T2_cw):
    """F such that x2^T F x1 = 0."""
    T21 = T2_cw @ inv_T(T1_cw)
    R, t = T21[:3, :3], T21[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Kinv = np.linalg.inv(K)
    return Kinv.T @ tx @ R @ Kinv


def epipolar_dist(F, x1, x2):
    """Distance in pixels from x2 to the epipolar line of x1."""
    x1h = np.hstack([x1, np.ones((len(x1), 1))])
    x2h = np.hstack([x2, np.ones((len(x2), 1))])
    l = x1h @ F.T
    return np.abs(np.sum(l * x2h, axis=1)) / (np.linalg.norm(l[:, :2], axis=1) + 1e-12)


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity (s, R, t) with dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t
