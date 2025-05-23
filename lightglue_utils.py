import cv2 
import torch
import numpy as np
import matplotlib.pyplot as plt

from lightglue import viz2d
from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import numpy_image_to_torch, rbd
from skimage.registration import optical_flow_tvl1, optical_flow_ilk

def compute_ssim():
    pass

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def pad_image(
    img: np.ndarray,
    pad: int | tuple[int, int, int, int] = 10,
    color: tuple[int, int, int] | int = (255, 255, 255),
) -> np.ndarray:
    # Resolve pad sizes
    if isinstance(pad, int):
        top = bottom = left = right = pad
    elif len(pad) == 4:
        top, bottom, left, right = pad
    else:
        raise ValueError("pad must be an int or a 4‑tuple (top, bottom, left, right)")

    # Convert colour for gray images
    if img.ndim == 2:                       # grayscale
        if isinstance(color, tuple):
            color = int(np.mean(color))
    else:                                   # colour image
        if not isinstance(color, tuple):
            color = (color, color, color)

    return cv2.copyMakeBorder(
        img,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

def clahe_lab(image, clipLimit=3.0, tileGridSize=(8, 8)):
    # Convert images to LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    # Split LAB channels
    l_image, a_image, b_image = cv2.split(image_lab)

    # Perform CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l_image_clahe = clahe.apply(l_image)

    # Merge channels back
    image_clahe = cv2.merge((l_image_clahe, a_image, b_image))

    # Convert back to BGR color space
    image_bgr = cv2.cvtColor(image_clahe, cv2.COLOR_Lab2RGB)

    return image_bgr

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

def glint_removal(img):
    h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    mask_glint = (v > 220) & (s < 30)
    img[mask_glint] = cv2.inpaint(img, mask_glint.astype('uint8')*255, 5, cv2.INPAINT_TELEA)

    return img

def zoom_image(zoom_factor, img_temp):
    # img_temp = image.copy()
    for i in range(2):
        img_temp = np.repeat(img_temp, zoom_factor, axis=i)

    return img_temp

class LGExtractor:

    def __init__(self, device="cpu"):
        self.device = device
        # self.extractor = SuperPoint(max_num_keypoints=4086, 
        #                             nms_radius=4, 
        #                             detection_threshold=0.005).eval().to(self.device)
        
        self.extractor = SIFT().eval().to(self.device)
        # self.extractor = DISK().eval().to(self.device)
        # self.extractor = SuperPoint().eval().to(self.device)
        self.matcher = LightGlue(features='sift', 
                                 max_kpts=4086, 
                                 filter_threshold=0.2, 
                                #  depth_confidence=0.3, 
                                #  width_confidence=0.3
                                 ).eval().to(self.device)

    def preprocess_image(self, image, blur=True, clahe=True, clahe_arguments=None):
        h,w,_ = image.shape
        # if min(h,w) < 250:
        #     image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        image = white_balance_lab(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = adjust_gamma(image, gamma=1.5)

        if blur:
            image = denoise_and_sharpen(image)
            # image = cv2.GaussianBlur(image, (5, 5), 0)

        # if clahe:
        #     if clahe_arguments is not None:
        #         image = clahe_lab(image, **clahe_arguments)
        #     else:
        #         image = clahe_lab(image)
        # image = glint_removal(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def extract_keypoints(self, image):
        return self.extractor.extract(image.to(self.device))

    def compute_match(self, feats0, feats1):
        return self.matcher({'image0': feats0, 'image1': feats1})
    
    def extract_and_match(self, image1, image2, component_type=None):
        master_component_clahe_args = {
            "clipLimit": 3.0,
            "tileGridSize": (8, 8)
        }

        sample_blister_clahe_args = {
            "clipLimit": 3.0,
            "tileGridSize": (8, 8)
        }

        # image1 = pad_image(image1, pad=10, color=(255, 255, 255))

        if component_type in ["logo", "warning_label"]:
            image1 = self.preprocess_image(image1, blur=True, clahe=True, clahe_arguments=master_component_clahe_args)
            image2 = self.preprocess_image(image2, blur=True, clahe=True)
        else:
            image1 = self.preprocess_image(image1, blur=True, clahe=True, clahe_arguments=master_component_clahe_args)
            image2 = self.preprocess_image(image2, blur=True, clahe=True, clahe_arguments=sample_blister_clahe_args)

        # h,w = image1.shape[:2]
        # image1 = cv2.resize(image1, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

        # image1,image2 = clahe_lab(image1, image2)
        # image1 = clahe_lab(image1)
        # image2 = clahe_lab(image2)
            
        # image1 = pad_image(image1, pad=30, color=(255, 255, 255))

        # zoom_image_factor = 4
        # image1 = zoom_image(zoom_image_factor, image1.copy())
        # image2 = zoom_image(zoom_image_factor, image2.copy())

        cv2.imwrite("image1.jpg", image1)
        cv2.imwrite("image2.jpg", image2)

        image1 = numpy_image_to_torch(image1)
        image2 = numpy_image_to_torch(image2)


        feats0 = self.extract_keypoints(image1)
        feats1 = self.extract_keypoints(image2)

        matches = self.compute_match(feats0, feats1)

        feats0, feats1, matches = [rbd(x) for x in [feats0, feats1, matches]]

        kpts0 = feats0["keypoints"]
        kpts1 = feats1["keypoints"]
        matches = matches['matches']  # indices with shape (K,2)
        points0 = kpts0[matches[..., 0]]  # coordinates in img0, shape (K,2)
        points1 = kpts1[matches[..., 1]]  # coordinates in img1, shape (K,2)

        return {
            "points0": points0.to("cpu"),
            "points1": points1.to("cpu"),
            # "matches01": matches01, 
            "matches": matches,
            "kpts0": kpts0.to("cpu"),
            "kpts1": kpts1.to("cpu"),
            "img0": image1,
            "img1": image2
        }
    
    def find_homography(self, src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0):
        homography, mask = cv2.findHomography(
            src_pts, 
            dst_pts, 
            method=method, 
            ransacReprojThreshold=ransacReprojThreshold
        )
        return homography, mask
    
    def crop_image(self, image, bbox, H):
        warped = cv2.perspectiveTransform(bbox[None], H)[0].astype(int)
        x,y,w,h = cv2.boundingRect(warped)
        crop    = image[y:y+h, x:x+w]

        return crop
    
    def identify_component(self, image1, image2, component_type=None):
        """
            Image1 = Master Component
            Image2 = Sample Blister 
        """


        # if component_type in ["logo", "warning_label", "composition", "salt_name", "mfg_details", "brand_logo", "label"]:
        #     print("Rotating Image")
        #     image1 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
        #     image2 = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)

        match_result = self.extract_and_match(image1, image2, component_type=component_type)

        # print(match_result)

        print(f"Number of Matches: {len(match_result['matches'])}")

        # print(match_result["points0"].numpy().reshape(-1, 1, 2))
        # print(match_result["points1"].numpy().reshape(-1, 1, 2))
        H, inliers = self.find_homography(
            match_result["points0"].numpy().reshape(-1, 1, 2), 
            match_result["points1"].numpy().reshape(-1, 1, 2), 
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=3.0
        )

        # H, m2 = cv2.estimateAffinePartial2D(match_result["points0"].numpy().reshape(-1, 1, 2), match_result["points1"].numpy().reshape(-1, 1, 2), method=cv2.RANSAC)
        inlier_count = np.sum(inliers) if inliers is not None else 0
        print(f"Homography inliers: {inlier_count}")

        bbox =  np.array([[0,0],[image1.shape[1],0],[image1.shape[1],image1.shape[0]],[0,image1.shape[0]]], dtype=np.float32)

        cropped_image = self.crop_image(image2, bbox, H)

        print(cropped_image.shape)

        return cropped_image



