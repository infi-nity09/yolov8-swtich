from fastapi import FastAPI, File, UploadFile
import torch, os, shutil
from ultralytics import YOLO
 
app = FastAPI()
model = YOLO("best.pt")  # Load the YOLOv8x model

def prediction(input_img_path):
    model.predict(input_img_path, save_crop=True, imgsz=640, conf=0.5)

    if os.path.exists("runs/detect/predict/crops/Negative"):
        return "Defective"
    return "Non Defective"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("temp.jpg", "wb") as f:  # Save temporarily
            f.write(contents)
 
        results = prediction('temp.jpg')  # Perform inference
 
        shutil.rmtree("runs/")
        return {"Prediction ":str(results)}
 
    except Exception as e:
        return {"error": str(e)}
 
    finally:
        try:
            os.remove("temp.jpg")  # Clean up
        except:
            pass