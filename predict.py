import os
import uuid
from ultralytics import YOLO
from PIL import Image

from app_config.config import MODEL_PATH


def infer_model(img_path):
    im1 = Image.open(img_path)
    load_model = YOLO(MODEL_PATH)
    results = load_model.predict(source=im1, save=True, imgsz=640, line_thickness=1, conf=0.25)[
        0]  # save plotted images

    return f"runs/detect/predict/{img_path.split('/')[-1]}"


if __name__ == '__main__':
    # img_path = "/home/shihab/Downloads/test_images/img_4.jpeg"
    img_path = "datatset/cat-dog-recongnition-1/test/images/depositphotos_75163555-stock-photo-cats-and-dogs-hanging-paws_jpg.rf.a7e5bb33979eb3d9d0982887e1620164.jpg"
    infer_model(img_path)
