"""Sparse bundle adjustment: Levenberg-Marquardt with Huber reweighting.

Poses are updated on the manifold, T <- exp(delta) * T with delta = (rotvec, translation),
and re-linearised at every iteration.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


def _skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1),
                     np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def bundle_adjust(K, poses_cw, points, cam_idx, pt_idx, obs, fixed_cams, max_iters=20, huber_px=2.0):
    """Refine camera poses and 3D points.

    poses_cw: (C,4,4) world->camera.  points: (P,3).
    cam_idx, pt_idx, obs: one row per observation (obs is Nx2 pixels).
    fixed_cams: boolean (C,), poses held constant (at least two, to fix the gauge).
    Returns (poses_cw, points, per-observation reprojection error in pixels).
    """
    R = np.array(poses_cw, float)[:, :3, :3]
    t = np.array(poses_cw, float)[:, :3, 3]
    X = np.array(points, float)
    cam_idx = np.asarray(cam_idx)
    pt_idx = np.asarray(pt_idx)
    fixed_cams = np.asarray(fixed_cams, bool)
    C, P, N = len(R), len(X), len(obs)
    free = np.flatnonzero(~fixed_cams)
    slot = -np.ones(C, int)
    slot[free] = np.arange(len(free))
    obs_slot = slot[cam_idx]
    m = obs_slot >= 0
    n_pose = 6 * len(free)
    n_var = n_pose + 3 * P
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    rows = np.broadcast_to(np.arange(2 * N).reshape(N, 2, 1), (N, 2, 3))
    pt_cols = np.broadcast_to((n_pose + 3 * pt_idx)[:, None, None] + np.arange(3), (N, 2, 3))
    rot_cols = np.broadcast_to((6 * obs_slot)[:, None, None] + np.arange(3), (N, 2, 3))
    r_idx = np.concatenate([rows.ravel(), rows[m].ravel(), rows[m].ravel()])
    c_idx = np.concatenate([pt_cols.ravel(), rot_cols[m].ravel(), (rot_cols[m] + 3).ravel()])

    def evaluate(R, t, X):
        Xc = np.einsum("nij,nj->ni", R[cam_idx], X[pt_idx]) + t[cam_idx]
        z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
        res = np.stack([fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy], 1) - obs
        return Xc, z, res

    def robust(res):
        e = np.linalg.norm(res, axis=1)
        w = np.where(e <= huber_px, 1.0, huber_px / np.maximum(e, 1e-12))
        cost = np.sum(np.where(e <= huber_px, 0.5 * e ** 2, huber_px * (e - 0.5 * huber_px)))
        return w, cost

    Xc, z, res = evaluate(R, t, X)
    w, cost = robust(res)
    lam = 1e-3
    for _ in range(max_iters):
        Jp = np.zeros((N, 2, 3))
        Jp[:, 0, 0] = fx / z
        Jp[:, 0, 2] = -fx * Xc[:, 0] / z ** 2
        Jp[:, 1, 1] = fy / z
        Jp[:, 1, 2] = -fy * Xc[:, 1] / z ** 2
        sw = np.sqrt(w)[:, None, None]
        J_pt = sw * (Jp @ R[cam_idx])
        J_rot = sw * (-Jp @ _skew(Xc))          # d(Xc)/d(rotvec) = -[Xc]x at delta = 0
        J_tr = sw * Jp
        vals = np.concatenate([J_pt.ravel(), J_rot[m].ravel(), J_tr[m].ravel()])
        J = coo_matrix((vals, (r_idx, c_idx)), shape=(2 * N, n_var)).tocsr()
        g = J.T @ (np.sqrt(w)[:, None] * res).ravel()
        # Schur complement: eliminate the (block-diagonal) point block, solve for poses densely.
        Jc, Jx = J[:, :n_pose], J[:, n_pose:]
        A = (Jc.T @ Jc).toarray()
        W = (Jc.T @ Jx).tocsr()
        V = np.zeros((P, 3, 3))
        np.add.at(V, pt_idx, np.einsum("nki,nkj->nij", J_pt, J_pt))
        gp, gx = g[:n_pose], g[n_pose:].reshape(P, 3)
        dA = np.maximum(np.diag(A), 1e-9)
        dV = np.maximum(np.einsum("pii->pi", V), 1e-9)
        blk = (3 * np.arange(P))[:, None, None]
        vi = np.broadcast_to(blk + np.arange(3)[None, :, None], (P, 3, 3)).ravel()
        vj = np.broadcast_to(blk + np.arange(3)[None, None, :], (P, 3, 3)).ravel()

        def solve(lam):
            Vl = V.copy()
            Vl[:, [0, 1, 2], [0, 1, 2]] += lam * dV
            Vinv = np.linalg.inv(Vl)
            Vinv_s = coo_matrix((Vinv.ravel(), (vi, vj)), shape=(3 * P, 3 * P)).tocsr()
            WV = W @ Vinv_s
            S = A + np.diag(lam * dA) - (WV @ W.T).toarray()
            dp = np.linalg.solve(S, -gp + WV @ gx.ravel()) if n_pose else np.zeros(0)
            dx = np.einsum("pij,pj->pi", Vinv, -gx - (W.T @ dp).reshape(P, 3))
            return np.concatenate([dp, dx.ravel()])

        improved = False
        for _ in range(8):
            try:
                step = solve(lam)
            except np.linalg.LinAlgError:
                step = np.full(n_var, np.nan)
            if not np.all(np.isfinite(step)):
                lam *= 10
                continue
            dp = step[:n_pose].reshape(-1, 6)
            dR = Rotation.from_rotvec(dp[:, :3]).as_matrix()
            R_new, t_new = R.copy(), t.copy()
            R_new[free] = dR @ R[free]
            t_new[free] = np.einsum("cij,cj->ci", dR, t[free]) + dp[:, 3:]
            X_new = X + step[n_pose:].reshape(P, 3)
            Xc_n, z_n, res_n = evaluate(R_new, t_new, X_new)
            w_n, cost_n = robust(res_n)
            if cost_n < cost:
                rel = (cost - cost_n) / max(cost, 1e-12)
                R, t, X, Xc, z, res, w, cost = R_new, t_new, X_new, Xc_n, z_n, res_n, w_n, cost_n
                lam = max(lam / 3, 1e-8)
                improved = True
                break
            lam *= 4
        if not improved or rel < 1e-6:
            break

    out = np.tile(np.eye(4), (C, 1, 1))
    out[:, :3, :3] = R
    out[:, :3, 3] = t
    return out, X, np.linalg.norm(res, axis=1)
