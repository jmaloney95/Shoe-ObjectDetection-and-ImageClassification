import cv2
import json
import os

from roboflow import Roboflow

# Object detection model
# rf = Roboflow(api_key="Fe80wL4p7rLnbPpfx6lI") - API key usage maxed
rf = Roboflow(api_key="Ph3dNUBinQO7VR7goYsB")
project = rf.workspace().project("shoe-object-detection-test")
model = project.version(2).model

# Image classification model
classification_project = rf.workspace().project("nike-adidas-and-converse-shoes-classification")
classification_model = classification_project.version(6).model

# List of image file names
input_folder = "input_images"  # Replace with the path to your folder containing the images
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Output directories for cropped regions and Converse images
cropped_output_dir = "cropped_regions"
converse_output_dir = "converse_images"

os.makedirs(cropped_output_dir, exist_ok=True)
os.makedirs(converse_output_dir, exist_ok=True)

# Function to process and save cropped regions for each image
def process_image(image_file):
    print(f"Processing {image_file}")

    result = model.predict(image_file, confidence=40, overlap=30)
    print("Prediction result:", result.json())

    image = cv2.imread(image_file)
    prediction_data = result.json()
    boxes = prediction_data['predictions']

    if not boxes:
        print(f"No objects detected in {image_file}")
        return

    # Find the highest confidence prediction
    highest_confidence_prediction = max(boxes, key=lambda x: x['confidence'])

    x1, y1 = int(highest_confidence_prediction['x']), int(highest_confidence_prediction['y'])
    x2, y2 = x1 + int(highest_confidence_prediction['width']), y1 + int(highest_confidence_prediction['height'])
    cropped_region = image[y1:y2, x1:x2]

    # Save the cropped region to a file with a unique name
    file_name = os.path.join(cropped_output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}_highest_confidence_cropped.jpeg")
    cv2.imwrite(file_name, cropped_region)
    print(f"Saved highest confidence cropped region for {image_file} as {file_name}")

    # Classify the cropped region
    classification_result = classification_model.predict(file_name).json()
    print(f"Classification result for {file_name}:", classification_result)

    top_prediction = classification_result['predictions'][0]['top']
    if top_prediction.lower() == 'converse':
        converse_image_path = os.path.join(converse_output_dir, os.path.basename(image_file))
        cv2.imwrite(converse_image_path, image)
        print(f"Saved original image with Converse detected as {converse_image_path}")

# Process and save cropped regions for all images in the list
for image_file in image_files:
    process_image(image_file)

input("Press Enter to exit...")