import torch
import cv2
import onnx
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ONNX model
onnx_model = onnx.load("model_- 28 april 2023 23_29.onnx")
ort_session = ort.InferenceSession(onnx_model.SerializeToString())

input_names = [inp.name for inp in ort_session.get_inputs()]

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

# Load and preprocess the input image
image_path = "c8.png"
input_batch = preprocess_image(image_path)

# Perform inference
output = ort_session.run(None, {'images': input_batch.numpy()})

# Process the output to obtain bounding boxes and class labels

def process_output(output, confidence_threshold=0.5):
    # Assuming the output is a list containing a single tensor of shape [1, num_detections, 6]
    # where the last dimension contains [class_id, confidence, x1, y1, x2, y2]
    # You might need to adjust this depending on your model's output format
    detections = output[0][0]

    # Convert the detections to the format used in the `process_image` function
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

    # Find the highest confidence prediction
    highest_confidence_prediction = max(boxes, key=lambda x: x['confidence'])

    return [highest_confidence_prediction]

# Draw the bounding boxes and save the result
image = cv2.imread(image_path)
for box in boxes:
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite("output.jpg", image)
