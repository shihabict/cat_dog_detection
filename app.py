import os
import shutil
from pathlib import Path
from starlette.background import BackgroundTask
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
from app_config.config import TEMP_IMG_DIR
from predict import infer_model

app = FastAPI(
    title="Object Detection",
    description="Cat and Dog Detection API",
    version="1.0.0"
)


def save_upload_file(upload_file: UploadFile, destination: Path):
    try:
        file_name = upload_file.filename
        test_filename = os.path.join(destination, file_name)
        with open(test_filename, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            file_name = buffer.name
            print(type(file_name))
    finally:
        upload_file.file.close()
    return file_name


def cleanup(predicted_img_path):
    predicted_img_dir = "/".join(predicted_img_path.split('/')[:-1])
    shutil.rmtree(predicted_img_dir)


@app.post("/predict/")
async def infer_trained_model(file: UploadFile = File(...)):
    allowed_formats = ["image/jpeg", "image/png"]
    if file.content_type not in allowed_formats:
        raise ValueError(f"File format {file.content_type} not supported")
    else:
        img_path = save_upload_file(file, TEMP_IMG_DIR)
        predicted_img_path = infer_model(img_path)
        os.remove(img_path)
        return FileResponse(predicted_img_path, media_type="image/jpeg",
                            background=BackgroundTask(cleanup, predicted_img_path))


@app.get("/")
async def main():
    return 'Object Detection API is working!'


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
