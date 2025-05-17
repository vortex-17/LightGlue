
from fastapi import FastAPI, Response
import uvicorn
from lightglue_utils import LGExtractor
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from http import HTTPStatus
import numpy as np
import cv2

app = FastAPI()

# Define Constants
LG_EXTRACTOR = LGExtractor(device="cpu")

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


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
    

@app.post("/identify_from_unique_id")
async def identify_from_unique_id(unqiue_id: str):
    pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
