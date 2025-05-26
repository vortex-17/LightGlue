
from fastapi import FastAPI, Response
import uvicorn
from lightglue_utils import LGExtractor
from services.image_services import get_master_components
from services.correction_utils import cropped_image
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from http import HTTPStatus
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import threading
import time

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Constants
LG_EXTRACTOR = LGExtractor(device="cpu")

@app.get("/healthcheck")
async def healthcheck():
    return JSONResponse(content={"status": "OK"}, status_code=200, media_type="application/json")


@app.post("/identify_single/{component_type}")
async def identify_single(component_type: str, master_component: UploadFile = File(...), sample_blister: UploadFile = File(...)):
    # try:
    image1_bytes = await master_component.read()
    nparr = np.frombuffer(image1_bytes, np.uint8)
    master_component_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image2_bytes = await sample_blister.read()
    nparr = np.frombuffer(image2_bytes, np.uint8)
    sample_blister_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    sample_component = LG_EXTRACTOR.identify_component(master_component_cv, sample_blister_cv, component_type)
    print("Sample Component: ", sample_component.shape)
    if 0 not in list(sample_component.shape):
        cv2.imwrite("sample_component.jpg", sample_component)
    return JSONResponse(content={"status": "success"}, media_type="application/json")
    # except Exception as e:
    #     print(f"Error while identifying component: {str(e)}")
    #     return JSONResponse(content={"status": "error", "message": str(e), "sample_component": None}, status_code=500, media_type="application/json"), 500

@app.post("/identify_components/{master_id}")
async def identify_components(master_id: str, sample_image: UploadFile = File(...)):
    try:
        image1_bytes = await sample_image.read()
        nparr = np.frombuffer(image1_bytes, np.uint8)
        sample_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Orient and Crop the image
        sample_image = cropped_image(sample_image)

    except Exception as e:
        print(f"Error while reading sample image: {str(e)}")

    def identify_component_thread(master_cmp, sample_image, master_component):
        sample_component = LG_EXTRACTOR.identify_component(master_cmp, sample_image, master_component)
        if sample_component is not None:
            # Compute MS-SSIM ?
            pass

    # Get all master components
    # Struct of Master Components -> {"logo" : cv2.imread(), "printed_details" : cv2.imread()}
    t1 = time.time()
    master_components = get_master_components(master_id)
    if not master_components:
        return JSONResponse(content={"status": "error", "message": "Master ID not found"}, status_code=404, media_type="application/json")
    
    print(f"Time taken to get master components: {time.time() - t1}")
    
    master_components_list = ["printed_details", "logo", "mfg_details", "warning_label", "salt_name", "composition", "label"]
    threads = []
    for master_component in master_components_list:
        print(f"Identifying component: {master_component}")
        master_cmp = master_components[master_component]
        thread = threading.Thread(target=identify_component_thread, args=(master_cmp, sample_image, master_component))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return JSONResponse(content={"status": "OK"}, status_code=200, media_type="application/json")
    

# Single Forward Pass for all components
@app.post("/identify_components/v3/{master_id}")
async def identify_components_v3(master_id: str, sample_image: UploadFile = File(...)):
    try:
        image1_bytes = await sample_image.read()
        nparr = np.frombuffer(image1_bytes, np.uint8)
        sample_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Orient and Crop the image
        sample_image = cropped_image(sample_image)

    except Exception as e:
        print(f"Error while reading sample image: {str(e)}")

    # Get Components
    LG_EXTRACTOR.identify_all_components_single_pass(sample_image, master_id)
    
    return JSONResponse(content={"status": "OK"}, status_code=200, media_type="application/json")
    


@app.post("/identify_from_unique_id")

async def identify_from_unique_id(unqiue_id: str):
    pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
