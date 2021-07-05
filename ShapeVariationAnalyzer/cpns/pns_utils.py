import numpy as np
def rotate_to_north_pole(v1, angle=None):
    """
    Rotate a unit vector v to the north pole of a unit sphere

    Return the rotation matrix
    See Rodrigues' rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    Input: 1D array (i.e., a point on a unit sphere)
    """
    d = len(v1) # dimension of the feature space

    ## normalize vector v
    v = v1 / np.linalg.norm(v1)
    ## north pole coordinates d-dimension [0, ..., 0, 1]
    north_pole = [0] * (d - 1)
    north_pole.append(1)
    north_pole = np.asarray(north_pole)

    inner_prod = np.inner(north_pole, v)
    if not angle:
        angle = np.arccos(inner_prod)
    if np.abs(inner_prod - 1) < 1e-15:
        return np.eye(d)
    elif np.abs(inner_prod + 1) < 1e-15:
        return -np.eye(d)
    c = v - north_pole * inner_prod
    c = c / np.linalg.norm(c)
    A = np.outer(north_pole, c) - np.outer(c, north_pole)

    rot = np.eye(d) + np.sin(angle)*A + (np.cos(angle) - 1)*(np.outer(north_pole, north_pole) + np.outer(c, c))
    return rot
def log_north_pole(x):
    """
    LOGNP Riemannian log map at North pole of S^k
        LogNP(x) returns k x n matrix where each column is a point on tangent
        space at north pole and the input x is (k+1) x n matrix where each column
        is a point on a sphere.
    Input: d x n matrix w.r.t. the extrinsic coords system
    Output: (d-1) x n matrix w.r.t. the coords system (tangent space) origined at the NP
    """
    d, n = x.shape
    scale = np.arccos(x[-1, :]) / np.sqrt(1-x[-1, :]**2)
    scale[np.isnan(scale)] = 1
    log_px = scale * x[:-1, :]
    return log_px
def exp_north_pole(x):
    """
    EXPNP Riemannian exponential map at North pole of S^k
    returns (k+1) x n matrix where each column is a point on a
    sphere and the input v is k x n matrix where each column
    is a point on tangent  space at north pole.

    Input: d x n matrix
    """
    d, n = x.shape
    nv = np.sqrt(np.sum(x ** 2, axis=0))
    tmp = np.sin(nv) / (nv + 1e-15)
    exp_px = np.vstack((tmp * x, np.cos(nv)))
    exp_px[:, nv < 1e-16] = np.repeat(np.vstack((np.zeros((d, 1)), 1)), np.sum(nv<1e-16), axis=1)
    return exp_px
