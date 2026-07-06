
import cv2
import numpy as np
import math
from typing import Literal, Tuple, Dict, Any

def _rotate_canvas(img: np.ndarray, ang_deg: float, border_value: int = 0) -> np.ndarray:
    """Rotate image by ang_deg onto the minimal canvas that contains it."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), ang_deg, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    new_w, new_h = int(h*sin + w*cos), int(h*cos + w*sin)
    M[0,2] += (new_w/2) - w/2
    M[1,2] += (new_h/2) - h/2
    flags = cv2.INTER_LINEAR if img.dtype != np.uint8 else cv2.INTER_NEAREST
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=flags,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border_value)

def _make_binary_mask(img_gray: np.ndarray, corridors_are_white: bool = True) -> np.ndarray:
    """Return a float32 0/1 mask where corridor pixels are 1."""
    blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not corridors_are_white:
        bw = 255 - bw
    return (bw > 0).astype(np.float32)

def _projection_variance(rot_mask: np.ndarray, orientation: str) -> float:
    """Variance of the 1-D projection along the chosen orientation."""
    if orientation == "horizontal":
        proj = rot_mask.sum(axis=1).astype(np.float64)  # per-row sum
    elif orientation == "vertical":
        proj = rot_mask.sum(axis=0).astype(np.float64)  # per-column sum
    else:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    proj -= proj.mean()
    return float((proj @ proj) / len(proj))

def _make_binary_mask(img_gray: np.ndarray, corridors_are_white: bool = True) -> np.ndarray:
    blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not corridors_are_white:
        bw = 255 - bw
    return (bw > 0).astype(np.uint8)   # <-- return uint8 mask (0/1)

def _objective(theta_deg: float, mask: np.ndarray, orientation: str,
               do_close: bool, close_frac: float) -> float:
    # Force NEAREST on masks by passing uint8
    rot = _rotate_canvas(mask.astype(np.uint8), theta_deg, border_value=0)
    if do_close:
        if orientation == "horizontal":
            dim = rot.shape[1] # width
            k = max(5, int(close_frac * dim))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        else:
            dim = rot.shape[0] # height
            k = max(5, int(close_frac * dim))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
        rot = cv2.morphologyEx(rot, cv2.MORPH_CLOSE, kernel, iterations=1)
        rot = (rot > 0).astype(np.float32)
    return _projection_variance(rot, orientation)

def _golden_section_max(f, a: float, b: float, tol: float = 0.01, max_iter: int = 200):
    """Golden-section maximization on [a,b]. Returns (theta, f(theta), iters)."""
    phi = (1 + 5 ** 0.5) / 2
    invphi = 1 / phi
    # interior points
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    fc = f(c); fd = f(d)
    it = 0
    while (b - a) > tol and it < max_iter:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = f(d)
        it += 1
    theta = 0.5 * (a + b)
    return theta, f(theta), it

def _quadratic_refine(f, theta: float, delta: float = 0.2) -> float:
    """Three-point quadratic interpolation refinement around theta."""
    t1, t2, t3 = theta - delta, theta, theta + delta
    J1, J2, J3 = f(t1), f(t2), f(t3)
    # Fit parabola J(t) = A t^2 + B t + C
    A = ((J3 - J2)/(t3 - t2) - (J2 - J1)/(t2 - t1)) / (t3 - t1 + 1e-12)
    B = (J2 - J1)/(t2 - t1 + 1e-12) - A*(t1 + t2)
    if abs(A) < 1e-12:
        return theta  # degenerate
    t_star = -B / (2*A)
    return t_star

def align_corridors(img_gray: np.ndarray,
                    orientation: Literal["horizontal","vertical"] = "horizontal",
                    corridors_are_white: bool = True,
                    downscale: float = 0.32,
                    tol_deg: float = 0.01,
                    do_close: bool = True,
                    close_frac: float = 0.015,
                    bg_value_fullres: int = 255) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """
    Find rotation that makes corridors as straight as possible.
    
    Args:
        img_gray: uint8 grayscale image.
        orientation: "horizontal" (rows) or "vertical" (columns) for the final corridor direction.
        corridors_are_white: set False if corridors are dark; mask will be inverted.
        downscale: factor for the search image (speed-up). Full-res is rotated once at the end.
        tol_deg: golden-section tolerance in degrees.
        do_close: apply 1D morphological closing along the corridor direction during scoring.
        close_frac: kernel length as a fraction of image width/height (orientation-dependent).
        bg_value_fullres: background fill for the *final* rotation of the grayscale image (e.g., 255 for white).
    
    Returns:
        angle_deg: best rotation angle (degrees).
        rotated_fullres: rotated grayscale image (full resolution).
        info: dict with keys {"J_best","theta_gs","iters","params"}.
    """
    assert img_gray.ndim == 2 and img_gray.dtype == np.uint8, "img_gray must be uint8 grayscale"
    # downscale for speed
    if downscale != 1.0:
        small = cv2.resize(img_gray, (0,0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    else:
        small = img_gray.copy()
    mask = _make_binary_mask(small, corridors_are_white=corridors_are_white)
    # objective closure
    def f(theta):
        return _objective(theta, mask, orientation=orientation, do_close=do_close, close_frac=close_frac)
    # golden-section search
    theta_gs, J_gs, iters = _golden_section_max(f, -90.0, 90.0, tol=tol_deg, max_iter=200)
    # local quadratic refinement
    theta_refined = _quadratic_refine(f, theta_gs, delta=0.2)
    J_best = f(theta_refined)
    # rotate full-res with the best angle
    rotated = _rotate_canvas(img_gray, theta_refined, border_value=bg_value_fullres)
    info = {
        "J_best": float(J_best),
        "theta_gs": float(theta_gs),
        "iters": int(iters),
        "params": {
            "orientation": orientation,
            "downscale": downscale,
            "tol_deg": tol_deg,
            "do_close": do_close,
            "close_frac": close_frac,
            "corridors_are_white": corridors_are_white,
        }
    }
    return float(theta_refined), rotated, info

#API
def estimate_corridor_angle(img_gray, **kwargs):
    angle, _rot, info = align_corridors(img_gray, **kwargs)
    return angle, info

# --- Optional CLI ---
def _cli():
    import argparse, sys
    ap = argparse.ArgumentParser(description="Align repeating corridor bands by rotation optimization.")
    ap.add_argument("input", help="input image (grayscale preferred)")
    ap.add_argument("-o","--output", default="aligned.png", help="output path")
    ap.add_argument("--orientation", choices=["horizontal","vertical"], default="horizontal")
    ap.add_argument("--dark", action="store_true", help="use if corridors are dark (invert mask)")
    ap.add_argument("--downscale", type=float, default=0.32)
    ap.add_argument("--tol", type=float, default=0.01, help="angle tolerance (degrees)")
    ap.add_argument("--no-close", action="store_true", help="disable 1D closing in objective")
    ap.add_argument("--bg", type=int, default=255, help="background value for final rotation (0-255)")
    args = ap.parse_args()
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read input:", args.input, file=sys.stderr); sys.exit(1)
    ang, rot, info = align_corridors(
        img, orientation=args.orientation,
        corridors_are_white=not args.dark,
        downscale=args.downscale,
        tol_deg=args.tol,
        do_close=(not args.no_close),
        bg_value_fullres=args.bg
    )
    cv2.imwrite(args.output, rot)
    print(f"Angle: {ang:.5f} deg")
    print(f"Saved: {args.output}")
    print("Info:", info)

if __name__ == "__main__":
    _cli()

