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
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

MONGO_URI = os.getenv("MONGO_URL")
PRESIGNED_S3_URL=os.getenv("S3_URL")

def get_image_from_s3(s3_link):
    url = f"{PRESIGNED_S3_URL}{s3_link}"
    resp = request("GET", url)
    if resp.status_code == 200:
        s3_presigned_url = str(resp.text).replace('"','')
        image = request("GET", s3_presigned_url, stream=True)
        if image.status_code == 200:
            # return image.text
            image_bytes = image.content
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image_data
    return None


def get_master_components(master_id):
    client = MongoClient(MONGO_URI)
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("MASTERS_COLLECTION_NAME")]
    component_dict = {}

    try:
        obj = collection.find_one({"_id" : ObjectId(master_id)})
        url_dict = obj["master_component_url"]
        for k,v in url_dict.items():
            print(k,v)
            image = get_image_from_s3(v)
            if image is not None:
                component_dict[k] = image
    
    except Exception as e:
        print(f"Error fetching master components from DB: {str(e)}")
    
    return component_dict