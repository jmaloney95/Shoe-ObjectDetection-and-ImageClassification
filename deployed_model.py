import torch
import cv2
import onnx
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ONNX model
onnx_model = onnx.load("model_- 28 april 2023 23_29.onnx")
ort_session = ort.InferenceSession(onnx_model.SerializeToString())

input_names = [inp.name for inp in ort_session.get_inputs()]

def preprocess_image(image_path):
    input_size = 640
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    return input_batch, preprocess

def save_cropped_region(image, box, output_path):
    x1, y1, x2, y2 = box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)

def process_output(output, confidence_threshold=0.5):
    detections = output[0][0]
    boxes = [
        {
            "class_id": int(detection[0]),
            "confidence": float(detection[1]),
            "x": int(detection[2]),
            "y": int(detection[3]),
            "width": abs(int(detection[4] - detection[2])),
            "height": abs(int(detection[5] - detection[3])),
        }
        for detection in detections
        if float(detection[1]) > confidence_threshold
    ]

    if not boxes:
        return []

    highest_confidence_prediction = max(boxes, key=lambda x: x['confidence'])

    print(highest_confidence_prediction)
    return [highest_confidence_prediction]
    

input_folder = "input_images"
output_folder = "cropped_regions"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    input_batch, preprocess = preprocess_image(image_path)

    output = ort_session.run(None, {'images': input_batch.numpy()})
    boxes = process_output(output)

    image = cv2.imread(image_path)
    for box in boxes:
        output_path = os.path.join(output_folder, f"cropped_{image_name}")
        save_cropped_region(image, box, output_path)
        print(f"Image: {image_name}, Bounding Box: x: {box['x']}, y: {box['y']}, width: {box['width']}, height: {box['height']}")  # Print bounding box dimensions
