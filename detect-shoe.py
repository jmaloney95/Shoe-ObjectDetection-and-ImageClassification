import os
import cv2
import onnx
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from roboflow import Roboflow
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ONNX model
onnx_model = onnx.load("model_- 28 april 2023 23_29.onnx")
ort_session = ort.InferenceSession(onnx_model.SerializeToString())

def preprocess_image(image_path):
    input_size = 640  # Replace with the required input size for your model
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    return input_batch

def process_output(output, confidence_threshold=0.5):
    detections = output[0][0]
    boxes = [
        {
            "class_id": int(detection[0]),
            "confidence": float(detection[1]),
            "x": int(detection[2]),
            "y": int(detection[3]),
            "width": int(detection[4] - detection[2]),
            "height": int(detection[5] - detection[3]),
        }
        for detection in detections
        if float(detection[1]) > confidence_threshold
    ]

    if not boxes:
        return []

    highest_confidence_prediction = max(boxes, key=lambda x: x['confidence'])
    return [highest_confidence_prediction]

# Replace with your Roboflow API key
rf = Roboflow(api_key="Fe80wL4p7rLnbPpfx6lI")
project = rf.workspace().project("nike-adidas-and-converse-shoes-classification")
classification_model = project.version(6).model

input_folder = "input_images"  # Replace with the path to your folder containing the images
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    input_batch = preprocess_image(image_file)
    output = ort_session.run(None, {'images': input_batch.numpy()})
    boxes = process_output(output)

    if not boxes:
        print(f"No objects detected in {image_file}")
        continue

    highest_confidence_prediction = boxes[0]
    image = cv2.imread(image_file)
    if image is None:
        print(f"Failed to load image: {image_file}")
        continue

    x1, y1 = int(highest_confidence_prediction['x']), int(highest_confidence_prediction['y'])
    x2, y2 = x1 + int(highest_confidence_prediction['width']), y1 + int(highest_confidence_prediction['height'])
    cropped_region = image[y1:y2, x1:x2]

    file_name = os.path.join("cropped_regions", f"{os.path.splitext(os.path.basename(image_file))[0]}_highest_confidence_cropped.jpeg")
    if cropped_region.size > 0:
        cv2.imwrite(file_name, cropped_region)
        print(f"Saved highest confidence cropped region for {image_file} as {file_name}")
    else:
        print(f"Failed to crop region for {image_file}")


    classification_result = classification_model.predict(file_name).json()
    top_prediction = classification_result['predictions'][0]['top']
    if top_prediction.lower() == 'converse':
        converse_image_path = os.path.join("converse_images", os.path.basename(image_file))
        cv2.imwrite(converse_image_path, image)
        print(f"Saved original image with Converse detected as {converse_image_path}")

# Process and save cropped regions for all images in the list
for image_file in image_files:
    process_output(image_file)

input("Press Enter to exit...")