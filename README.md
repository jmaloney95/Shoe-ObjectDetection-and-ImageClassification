# Shoe Detection and Classification

This project is focused on detecting shoes in images and classifying them using object detection and image classification models. The object detection model is trained to detect shoes in images, while the image classification model is trained to classify the detected shoes as Converse, Nike, or Adidas.

## Prerequisites

- Python 3.7+
- OpenCV
- PyTorch
- torchvision
- Roboflow

You can install the required Python libraries using pip:

```bash
pip install opencv-python torch torchvision roboflow
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your_username/shoe-detection-and-classification.git
cd shoe-detection-and-classification
```

2. Update the following variables in the code to match your setup:

- Your Roboflow API key
- Paths to the object detection and image classification model definition files and weights
- Paths to input images or input image folder
- Required input size for your models

3. Run the main script:

```bash
python main.py
```

The script will perform the following tasks:

- Process each image in the input folder
- Use the object detection model to detect shoes in the images
- Crop the highest-confidence shoe detection from each image
- Use the image classification model to classify the cropped shoe as Converse, Nike, or Adidas
- If the classification is 'Converse', save the original image with the detected Converse shoe to a separate folder

## Customizing the Project

You can easily modify this project to detect and classify other objects by:

1. Training new object detection and image classification models for your specific use case.
2. Updating the object detection and image classification model paths in the code.
3. Modifying the preprocessing steps and output processing functions to match the requirements of your models.

## License

This project is released under the MIT License. See the [LICENSE](https://opensource.org/license/mit/) file for more information.
