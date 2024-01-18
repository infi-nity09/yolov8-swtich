'''
This API is used for comparing images belonging to multiple classes and finding out defects. Steps are as follows:
  1. Upload the input image to the API. Along with it pass info about the type of image i.e grid or layout or power_eco, etc.,
  2. Image of the correponding type i.e not defective img will be fetched from local
  3. Siamese Network compares the difference between the images
  4a. If the difference/distance is very small then the input image should also be similar to local image 
      i.e the component in input image should also most probably not be defective
  4b. If the difference/distance is large then the input image is not similar to the local image
      i.e the component in input image should be defective
'''
import os, shutil
from ultralytics import YOLO
from pydantic import BaseModel, validator
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException

app = FastAPI()

def prediction(input_img_path):
    model = YOLO('yolov8x_v2.pt')
    model.predict(input_img_path, save_crop=True, imgsz=640, conf=0.5)

    if os.path.exists("runs/detect/predict/crops/Negative"):
        return "Defective"
    return "Non Defective"

class ImageInput(BaseModel):
    file: UploadFile = File(...)
    input_text: str = Form(...)

    @validator("input_text")
    def validate_input_text(cls, value):
        # allowed_values = {'ps_lights', 'ps_fe', 'ps_grid', 'ps_pe', 'ps_four', 'ps_top', 'dashboard', 'dup', 'layout', 'grid', 'two'}
        allowed_value = {'grid', 'top', 'fe'}
        if value.lower() not in allowed_value:
            # error = "Input should be one among the following ['ps_lights', 'ps_fe', 'ps_grid', 'ps_pe', 'ps_four', 'ps_top', 'dashboard', 'dup', 'layout', 'grid', 'two']"
            error = "Input should be one among the following ['grid', 'top', 'fe']"
            raise HTTPException(status_code=422, detail=str(error))
        return value.lower()

def perform_validation(image_input: ImageInput = Depends()):
    return image_input

@app.post("/upload/")
async def upload_image(image_input: ImageInput = Depends(perform_validation)):
    # Here, 'file' is the uploaded image.

    # You can save the uploaded file to a directory.
    with open(f"uploaded_images/{image_input.file.filename}", "wb") as f:
        f.write(image_input.file.file.read())

    input_img_path = f"uploaded_images/{image_input.file.filename}"
    results = prediction(input_img_path)

    
    shutil.rmtree("runs/")

    #return {"filename": file.filename}
    return {"filename": image_input.file.filename, "category":image_input.input_text , "prediction":str(results)}
