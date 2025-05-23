import cv2 
import torch
import numpy as np
import matplotlib.pyplot as plt

from lightglue import viz2d
from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import numpy_image_to_torch, rbd
from skimage.registration import optical_flow_tvl1, optical_flow_ilk

from img_preprocessing_utils import denoise_and_sharpen, white_balance_lab, refine_homography_ecc, tight_crop_border,contour_trim
class LGExtractor:

    def __init__(self, device="cpu"):
        self.device = device
        # self.extractor = SuperPoint(max_num_keypoints=4086, 
        #                             nms_radius=4, 
        #                             detection_threshold=0.005).eval().to(self.device)
        
        self.extractor = SIFT().eval().to(self.device)
        self.matcher = LightGlue(features='sift', 
                                 max_kpts=4086, 
                                 filter_threshold=0.2, 
                                #  depth_confidence=0.3, 
                                #  width_confidence=0.3
                                 ).eval().to(self.device)

    def preprocess_image(self, image, blur=True):
        image = white_balance_lab(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if blur:
            image = denoise_and_sharpen(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def extract_keypoints(self, image):
        return self.extractor.extract(image.to(self.device))

    def compute_match(self, feats0, feats1):
        return self.matcher({'image0': feats0, 'image1': feats1})
    
    def extract_and_match(self, image1, image2, component_type=None):
    
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)

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

        print(f"Number of Matches: {len(match_result['matches'])}")

        H, inliers = self.find_homography(
            match_result["points0"].numpy().reshape(-1, 1, 2), 
            match_result["points1"].numpy().reshape(-1, 1, 2), 
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=3.0
        )

        # H1 = refine_homography_ecc(image1, image2, H)

        # H, m2 = cv2.estimateAffinePartial2D(match_result["points0"].numpy().reshape(-1, 1, 2), match_result["points1"].numpy().reshape(-1, 1, 2), method=cv2.RANSAC)
        inlier_count = np.sum(inliers) if inliers is not None else 0
        print(f"Homography inliers: {inlier_count}")

        bbox =  np.array([[0,0],[image1.shape[1],0],[image1.shape[1],image1.shape[0]],[0,image1.shape[0]]], dtype=np.float32)

        cropped_image = self.crop_image(image2, bbox, H)

        print(cropped_image.shape)

        if component_type in ["warning_label", "logo"]:
            cropped_image = tight_crop_border(cropped_image, bg_threshold=180)
            cropped_image = contour_trim(cropped_image)

        return cropped_image



