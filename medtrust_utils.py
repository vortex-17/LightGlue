# This file contains MedTrust Utils functions for testing

import numpy as np
import time
import pandas as pd
import json
import os
from requests import request
from pymongo import MongoClient
from bson.objectid import ObjectId
import shutil
from uuid import uuid4
import base64
from dotenv import load_dotenv
import os
import cv2
from lightglue_utils import LGExtractor

LG_EXTRACTOR = LGExtractor(device="cpu")

load_dotenv()

SKU_MASTER_ID_MAP = {
    "67ecd2ae7ae3dc209c80bc0e" : "LEVIPILL DEMO",
    "67ecd4427ae3dc209c80bc0f" : "MONTEK LC DEMO",
    "67e678ee7ae3dc209c80bc0b" : "MONTEK LC",
    "6825d80309398f7f7c1ffadc" : "ZYDUS NUCOXIA MR",
    "6825da3e09398f7f7c1ffadd" : "DECA DUROBOLIN 1",
    "6825de5209398f7f7c1ffade" : "DECA DUROBOLIN 2",
    "6826149009398f7f7c1ffadf" : "ATORVA"
}

# Constants
IMAGE_FOLDER = os.path.join(os.getcwd(),"images")
MASTER_IMAGES = f"{IMAGE_FOLDER}/master/"
COMPONENTS_IMAGES = f"{IMAGE_FOLDER}/components/"
SUBFOLDER = "medtrust"
MONGO_URI = os.getenv("MONGO_URL")
PRESIGNED_S3_URL=os.getenv("S3_URL")


def get_image_from_s3(s3_link, master_id, image_path=None):
    url = f"{PRESIGNED_S3_URL}{s3_link}"
    resp = request("GET", url)
    if resp.status_code == 200:
        s3_presigned_url = str(resp.text).replace('"','')
        image = request("GET", s3_presigned_url, stream=True)
        if image.status_code == 200:
            if image_path is None:
                image_path = f"images/{SUBFOLDER}/{str(uuid4())}_{master_id}.jpeg"
            with open(image_path, 'wb') as out_file:
                shutil.copyfileobj(image.raw, out_file)
                
            return image_path
    return None

def get_master_components_s3_links(master_id):
    client = MongoClient(MONGO_URI)
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("MASTERS_COLLECTION_NAME")]

    try:
        obj = collection.find_one({"_id" : ObjectId(master_id)})
        url_dict = obj["master_component_url"]
        for k,v in url_dict.items():
            image_path = f"{MASTER_IMAGES}/{master_id}/"
            if not os.path.exists(image_path):
                os.makedirs(image_path, exist_ok=True)

            image_path += f"{k}.jpeg"
            get_image_from_s3(v, master_id, image_path)
    
    except Exception as e:
        print(f"Error fetching master components from DB: {str(e)}")
        return None



def check_master_component_images(master_id):
    path = os.path.join(MASTER_IMAGES, master_id)
    if os.path.exists(path):
        files = os.listdir(path)
        if len(files) == 0:
            return False
        return True
    
    return False

def get_sample_blister_image(unique_id):
    # print(MONGO_URI)
    client = MongoClient(MONGO_URI)
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("RESULTS_COLLECTION_NAME")]

    # print(collection.find_one())

    try:
        obj = collection.find_one({"unique_id" : unique_id})
        master_id = obj["master_id"]
        sample_blister_url = obj["corrected_sample_image_url"]
        image_path = f"{IMAGE_FOLDER}/sample_blisters/{unique_id}.jpeg"
        get_image_from_s3(sample_blister_url, unique_id, image_path)
        return image_path, master_id
    except Exception as e:
        print(f"Error fetching sample blister from DB: {str(e)}")
        return None, None
    


def get_components(unique_id):
    sample_image_path, master_id = get_sample_blister_image(unique_id)
    if master_id:
        if not check_master_component_images(master_id):
            get_master_components_s3_links(master_id)
        else:
            print("Master components already downloaded.")

    print(sample_image_path)

    components = ["logo", "printed_details", "brand_logo", "warning_label", "composition", "salt_name", "mfg_details", "label"]
    final_component_dict = {
        "master_id" : master_id,
        "SKU" : SKU_MASTER_ID_MAP.get(master_id, None),
    }
    component_dict = {}

    for component in components:
        print(f"Finding Component {component}")
        component_dict[component] = True
        master_components_path = os.path.join(MASTER_IMAGES, master_id)
        component_path = f"{master_components_path}/{component}.jpeg"
        if not os.path.exists(component_path):
            component_dict[component] = None
            continue
        master_component = cv2.imread(f"{master_components_path}/{component}.jpeg")
        sample_blister = cv2.imread(sample_image_path)
        
        try:
            sample_component = LG_EXTRACTOR.identify_component(master_component, sample_blister, None)

            sample_component_path = f"{COMPONENTS_IMAGES}/{unique_id}"
            if not os.path.exists(sample_component_path):
                os.makedirs(sample_component_path, exist_ok=True)

            cv2.imwrite(f"{sample_component_path}/{component}.jpeg", sample_component)
        except Exception as e:
            print(f"Error while identifying component: {str(e)}")
            component_dict[component] = {
                "status" : "error",
                "error" : str(e)
            }
    final_component_dict["components"] = component_dict
    return final_component_dict


def get_list_of_ids(file_path):
    data = pd.read_csv(file_path)
    if data.empty:
        return []
    
    return data["Unique ID"].tolist()

def main(log_results=True):
    t0 = time.time()
    unique_ids = []
    file_path = "/Users/vivek/allscan/counterfeit_demo/demo_11042025_fake.csv"
    unique_ids += get_list_of_ids(file_path)
    err_dict = {}
    completed_dict = {}
    unique_ids = list(set(unique_ids))
    print(f"Total Unique IDs: {len(unique_ids)}")
    for unique_id in unique_ids:
        print(unique_id)
        try:
            t1 = time.time()
            component_dict = get_components(unique_id)
            if component_dict:
                completed_dict[unique_id] = component_dict
            else:
                err_dict[unique_id] = "No components found"
            print(f"Time taken for {unique_id}: {time.time() - t1}")
        except Exception as e:
            print(f"Error processing unique_id {unique_id}: {str(e)}")
            err_dict[unique_id] = str(e)
            continue

    print(f"Total Time taken for {len(unique_ids)} blisters: {time.time() - t0}")

    if log_results:
        with open("error_list.json", "a") as fp:
            fp.write(json.dumps(err_dict, indent=4))

        with open("final_result.json", "a") as fp:
            fp.write(json.dumps(completed_dict, indent=4))

    

if __name__ == "__main__":
    main(log_results=False)