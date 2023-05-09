from ultralytics import YOLO
import cv2

def save_cropped_region(image, box, output_path):
    x1, y1, x2, y2 = box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)

model = YOLO("model.pt")

# Use the appropriate source for your image
source_image_path = "image_path"
results = model.predict(source=source_image_path)

image = cv2.imread(source_image_path)
for result in results:
    boxes = result.boxes.xyxy
    confs = result.boxes.conf
    clss = result.boxes.cls
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        conf, cls = confs[i], clss[i]
        output_path = "cropped_output.jpg"  # Specify the output path for the cropped image
        save_cropped_region(image, {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}, output_path)
