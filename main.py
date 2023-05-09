import os
import cv2
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from roboflow import Roboflow
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_cropped_region(image, box, output_path):
    x1, y1, x2, y2 = box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)

model = YOLO("best.pt")

# Replace with your Roboflow API key
rf = Roboflow(api_key="Fe80wL4p7rLnbPpfx6lI")
project = rf.workspace().project("nike-adidas-and-converse-shoes-classification")
classification_model = project.version(6).model

input_folder = "input_images"
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

cropped_output_dir = "cropped_regions"
converse_output_dir = "converse_images"

os.makedirs(cropped_output_dir, exist_ok=True)
os.makedirs(converse_output_dir, exist_ok=True)

confidence_threshold = 0.5

for image_file in image_files:
    try:
        results = model.predict(source=image_file)
    except FileNotFoundError:
        print(f"Image not found: {image_file}")
        continue

    results = model.predict(source=image_file)
    image = cv2.imread(image_file)
    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        clss = result.boxes.cls
        for i in range(len(boxes)):
            conf = confs[i]
            if conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, boxes[i])
                cls = clss[i]
                output_path = os.path.join(cropped_output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}_{cls}_{conf:.2f}_cropped.jpeg")
                save_cropped_region(image, {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}, output_path)
                print(f"Saved cropped region with {conf:.2f} confidence for {image_file} as {output_path}")

                classification_result = classification_model.predict(output_path).json()
                top_prediction = classification_result['predictions'][0]['top']
                if top_prediction.lower() == 'converse':
                    converse_image_path = os.path.join(converse_output_dir, os.path.basename(image_file))
                    cv2.imwrite(converse_image_path, image)
                    print(f"Saved original image with Converse detected as {converse_image_path}")

input("Press Enter to exit...")
