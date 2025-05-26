import cv2 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from lightglue import viz2d
from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import numpy_image_to_torch, rbd
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from services.img_preprocessing_utils import denoise_and_sharpen, white_balance_lab, refine_homography_ecc, tight_crop_border,contour_trim, get_correct_orientation_and_skew

load_dotenv()

IMAGE_FOLDER = os.path.join(os.getcwd(),"images")
MASTER_IMAGES = f"{IMAGE_FOLDER}/master/"
COMPONENTS_IMAGES = f"{IMAGE_FOLDER}/components/"
SUBFOLDER = "medtrust"
MONGO_URI = os.getenv("MONGO_URL")
PRESIGNED_S3_URL=os.getenv("S3_URL")

class BatchedLG(nn.Module):
    def __init__(self, lg):
        super().__init__()
        self.lg = lg
    def forward(self, pair_list):
        outs = []
        for p in pair_list:                 # stays in C++, GIL released
            outs.append(self.lg(p))
        return outs

class LGExtractor:

    def __init__(self, device="cpu"):
        self.device = device
        # self.extractor = SuperPoint(max_num_keypoints=4086, 
        #                             nms_radius=4, 
        #                             detection_threshold=0.005).eval().to(self.device)
        
        self.extractor = SIFT().eval().to(self.device)
        self.matcher = LightGlue(features='sift', 
                                 max_kpts=1000, 
                                 filter_threshold=0.2, 
                                #  depth_confidence=0.3, 
                                #  width_confidence=0.3
                                 ).eval().to(self.device).to(torch.float32)
        
        # self.matcher.compile(mode='reduce-overhead')

        self.batch_matcher = BatchedLG(self.matcher)

    def preprocess_image(self, image, blur=True):
        image = white_balance_lab(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if blur:
            image = denoise_and_sharpen(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    @torch.no_grad()
    def extract_keypoints(self, image):
        image = self.preprocess_image(image)
        image = numpy_image_to_torch(image)
        features = self.extractor.extract(image.to(self.device))
        return features

    @torch.no_grad()
    def compute_match(self, feats0, feats1):
        return self.matcher({'image0': feats0, 'image1': feats1})
    
    def extract_and_match(self, image1, image2, component_type=None):

        t1 = time.time()
        feats0 = self.extract_keypoints(image1)
        feats1 = self.extract_keypoints(image2)
        print(f"Keypoint Extraction Time: {time.time() - t1:.2f} seconds")

        t1 = time.time()
        matches = self.compute_match(feats0, feats1)
        print(f"Matching Time: {time.time() - t1:.2f} seconds")

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

        print(image1.shape)
        print(image2.shape)

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
        
        rotate = True
        if component_type == "printed_details":
            rotate = False
        
        cropped_image = get_correct_orientation_and_skew(
                cropped_image, rotate
            )

        if component_type in ["warning_label", "logo"]:
            cropped_image = tight_crop_border(cropped_image, bg_threshold=180)
            cropped_image = contour_trim(cropped_image)

        return cropped_image


    # WIP - Trying to find a way to do a single forward pass on the LightGlue Match
    def identify_all_components_single_pass(self, sample_image, master_id):

        # fetch all master_id
        component_list = ["logo", "mfg_details", "warning_label", "printed_details", "label","salt_name","composition"]
        component_dict = {}
        
        sample_image_feature = self.extract_keypoints(sample_image)
        pair_list = []
        for component in component_list:
            print(component)
            component_path = f"{MASTER_IMAGES}/{master_id}/{component}.jpeg"
            if os.path.exists(component_path):
                master_component = cv2.imread(component_path)
                master_component_features = self.extract_keypoints(master_component)
                component_dict[component] = {
                    "image" : master_component,
                    "features" : master_component_features
                }
                pair_list.append({"image0" : master_component_features, "image1" : sample_image_feature})

        print(component_dict)
            
        t1 = time.time()
        matches_all = self.batch_matcher(pair_list)
        

        for i in range(len(component_list)):
            try:
                component = component_list[i]
                match_result = matches_all[i]
                feats0, feats1, match_result = [rbd(x) for x in [component_dict[component_list[i]]["features"], sample_image_feature, match_result]]
                print(match_result.keys())
                kpts0 = feats0["keypoints"].to("cpu")
                kpts1 = feats1["keypoints"].to("cpu")
                matches = match_result['matches']  # indices with shape (K,2)
                points0 = kpts0[matches[..., 0]].to("cpu")  # coordinates in img0, shape (K,2)
                points1 = kpts1[matches[..., 1]].to("cpu") # coordinates in img1, shape (K,2)
                H, inliers = self.find_homography(
                    points0.numpy().reshape(-1, 1, 2), 
                    points1.numpy().reshape(-1, 1, 2), 
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=3.0
                )

                # H, m2 = cv2.estimateAffinePartial2D(match_result["points0"].numpy().reshape(-1, 1, 2), match_result["points1"].numpy().reshape(-1, 1, 2), method=cv2.RANSAC)
                inlier_count = np.sum(inliers) if inliers is not None else 0
                print(f"Homography inliers: {inlier_count}")

                image1 = component_dict[component_list[i]]["image"]

                bbox =  np.array([[0,0],[image1.shape[1],0],[image1.shape[1],image1.shape[0]],[0,image1.shape[0]]], dtype=np.float32)

                cropped_image = self.crop_image(sample_image, bbox, H)
                if component == "printed_details":
                    rotate = False
                
                cropped_image = get_correct_orientation_and_skew(
                        cropped_image, rotate
                    )

                if component in ["warning_label", "logo"]:
                    cropped_image = tight_crop_border(cropped_image, bg_threshold=180)
                    cropped_image = contour_trim(cropped_image)
                
                cv2.imwrite(f"{component}.jpeg", cropped_image)
            except Exception as e:
                print(f"Error in component {component_list[i]}: {str(e)}")
                cropped_image = None



        print(f"Time taken for matching: {time.time() - t1}")
        



# t0 = time.time()
# sample_image = cv2.imread(f"{IMAGE_FOLDER}/sample_blisters/d335909f-cb77-44db-a6bd-d678ea484528.jpeg")
# master_id = "67ecd2ae7ae3dc209c80bc0e"

# lg = LGExtractor(device="cpu")
# for i in range(1):
#     lg.identify_all_components_single_pass(sample_image, master_id)

# print(f"Total Time taken: {time.time() - t0}")