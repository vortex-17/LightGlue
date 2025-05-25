import cv2 
import torch
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

from img_preprocessing_utils import denoise_and_sharpen, white_balance_lab, refine_homography_ecc, tight_crop_border,contour_trim, get_correct_orientation_and_skew

load_dotenv()

IMAGE_FOLDER = os.path.join(os.getcwd(),"images")
MASTER_IMAGES = f"{IMAGE_FOLDER}/master/"
COMPONENTS_IMAGES = f"{IMAGE_FOLDER}/components/"
SUBFOLDER = "medtrust"
MONGO_URI = os.getenv("MONGO_URL")
PRESIGNED_S3_URL=os.getenv("S3_URL")

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
                                 ).eval().to(self.device)
        
        self.matcher.compile(mode='reduce-overhead')

    def preprocess_image(self, image, blur=True):
        image = white_balance_lab(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if blur:
            image = denoise_and_sharpen(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    @torch.no_grad()
    def extract_keypoints(self, image):
        features = self.extractor.extract(image.to(self.device))
        return features

    @torch.no_grad()
    def compute_match(self, feats0, feats1):
        return self.matcher({'image0': feats0, 'image1': feats1})
    
    def extract_and_match(self, image1, image2, component_type=None):
    
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)

        cv2.imwrite("image1.jpg", image1)
        cv2.imwrite("image2.jpg", image2)

        image1 = numpy_image_to_torch(image1)
        image2 = numpy_image_to_torch(image2)

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
        component_list = ["logo", "mfg_details", "warning_label", "printed_details"]
        component_dict = {}
        master_features = []
        offsets = []
        cursor = 0
        for component in component_list:
            print(component)
            component_path = f"{MASTER_IMAGES}/{master_id}/{component}.jpeg"
            if os.path.exists(component_path):
                master_component = cv2.imread(component_path)
                print(master_component.shape)
                preprocessed_image = self.preprocess_image(master_component, blur=True)
                numpy_comp = numpy_image_to_torch(preprocessed_image)

                master_component_features = self.extract_keypoints(numpy_comp)

                # print(type(master_component_features["keypoints"]))
                
                # Keys : 'keypoints', 'scales', 'oris', 'descriptors', 'keypoint_scores', 'image_size'
                keypoints = master_component_features['keypoints']
                scales = master_component_features['scales']
                oris = master_component_features['oris']
                descriptors = master_component_features['descriptors']
                size_t = torch.tensor(master_component.shape[::-1], device=self.device)

                master_features.append({"keypoints": keypoints, "scales": scales, "oris": oris, "descriptors": descriptors, "image_size": size_t})
                offsets.append((cursor, cursor + keypoints.shape[0]))
                cursor += keypoints.shape[0]

        for mf in master_features:
            print(mf["keypoints"].shape)

        key_cat   = torch.cat([mf["keypoints"] for mf in master_features if mf], dim=1)
        desc_cat  = torch.cat([mf["descriptors"] for mf in master_features if mf], dim=1)
        oris_cat = torch.cat([mf["oris"] for mf in master_features if mf], dim=1)
        scales_cat = torch.cat([mf["scales"] for mf in master_features if mf], dim=1)

        size_cat  = torch.tensor([[sample_image.shape[1], sample_image.shape[0]]], device=self.device)

        feats_master = {"keypoints": key_cat, "descriptors": desc_cat, "oris": oris_cat, "scales": scales_cat, "image_size": size_cat}
        
        sample_blister_processed = self.preprocess_image(sample_image)
        sample_blister_features = self.extract_keypoints(numpy_image_to_torch(sample_blister_processed))

        t1 = time.time()
        matches = self.matcher({
            "image0": sample_blister_features,
            "image1": feats_master
        })

        print(key_cat.shape)

        print(f"Time taken for feature matching: {time.time() - t1}")

        feats0, feats1, matches = [rbd(x) for x in [sample_blister_features, feats_master, matches]]
        kpts0 = feats0["keypoints"].cpu().numpy()
        matches_np = matches["matches"].cpu().numpy()

        print(matches_np.shape)

        per_comp_pairs = [[] for _ in component_list]
        for idx0, idx1 in enumerate(matches_np):
            # print(idx0.shape)
            print(idx1)
            # if idx1 < 0:
            #     continue
            for ci, (lo, hi) in enumerate(offsets):
                if lo <= idx1.any() < hi:
                    per_comp_pairs[ci].append((idx0, idx1 - lo))
                    break

        print(per_comp_pairs)
        print(len(per_comp_pairs))

        def _process_ci(ci):
            mf = master_features[ci]
            if mf is None or len(per_comp_pairs[ci]) < 4:
                return None
            pts0 = np.float32([kpts0[i] for i, _ in per_comp_pairs[ci]])
            pts1_m = mf["keypoints"].cpu().numpy()
            pts1 = np.float32([pts1_m[j] for _, j in per_comp_pairs[ci]])
            H, inl = cv2.findHomography(pts1, pts0, cv2.USAC_MAGSAC, 4.0)
            if H is None:
                return None
            w_m, h_m = mf["image_size"].cpu().numpy()[0]
            bbox = np.array([[0, 0], [w_m, 0], [w_m, h_m], [0, h_m]], np.float32)
            crop = self.crop_image(sample_image, bbox, H)
            rotate = component_list[ci] != "printed_details"
            crop = get_correct_orientation_and_skew(crop, rotate)
            if component_list[ci] in ["warning_label", "logo"]:
                crop = tight_crop_border(crop, bg_threshold=180)
                crop = contour_trim(crop)
            return crop

        results = {}
        with ThreadPoolExecutor(max_workers=min(len(component_list), os.cpu_count() or 1)) as pool:
            for ci, crop in enumerate(pool.map(_process_ci, range(len(component_list)))):
                results[component_list[ci]] = crop
        return results


        




# sample_image = cv2.imread(f"{IMAGE_FOLDER}/sample_blisters/42fe10fc-5870-47fc-876d-7185bd0c29b0.jpeg")
# master_id = "67ecd2ae7ae3dc209c80bc0e"

# lg = LGExtractor(device="cpu")
# lg.identify_all_components_test(sample_image, master_id)