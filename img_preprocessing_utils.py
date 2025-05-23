import cv2
import numpy as np

def denoise_and_sharpen(img):
    # img = cv2.bilateralFilter(img, 5, 75, 75)
    img = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.15)
    # img   = cv2.GaussianBlur(img, (5,5), 2.0)
    # sharp  = cv2.addWeighted(smooth, 1.7, blur, -0.7, 0)
    return img

def white_balance_lab(img):
    # Convert BGR → Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # Compute average a/b (128 = neutral)
    a_avg = np.mean(a)
    b_avg = np.mean(b)

    # Subtract a/b offset, scaled by luminance
    a -= (a_avg - 128) * (l / 255.0)
    b -= (b_avg - 128) * (l / 255.0)

    # Re-merge & convert back to BGR
    lab = cv2.merge([l, a, b])
    balanced = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return balanced

def refine_homography_ecc(master_roi, sample_roi, H0):
    # warp master patch into sample space with current guess
    h, w = master_roi.shape[:2]
    cv2.imshow("master_roi", master_roi)
    cv2.imshow("sample_roi", sample_roi)
    cv2.waitKey(0)
    warp = cv2.warpPerspective(master_roi, H0, (w, h))
    
    # initialise affine (2×3) from 3×3 H0
    M0 = H0[:2, :3].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
    cc, M = cv2.findTransformECC(
        cv2.cvtColor(sample_roi, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(warp,        cv2.COLOR_BGR2GRAY),
        M0, cv2.MOTION_AFFINE, criteria,
    )
    # upgrade affine → 3×3 so you can reuse cv2.warpPerspective
    H_refined           = np.eye(3, dtype=np.float32)
    H_refined[:2, :3]   = M
    return H_refined @ H0

def correct_component_skew():
    pass

def tight_crop_border(img, bg_threshold=240):
    """Remove near-white or transparent border from RGBA / BGR image."""
    if img.shape[2] == 4:           # alpha present → just mask on alpha
        mask = img[..., 3] > 0
    else:                           # use near-white test in BGR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray < bg_threshold
    
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:            # completely blank (safety guard)
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def contour_trim(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    th, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    
    # remove small specks
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    
    # merge all contours into one bounding box
    x,y,w,h = cv2.boundingRect(np.vstack(cnts))
    return img[y:y+h, x:x+w]