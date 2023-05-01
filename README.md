<img src = https://user-images.githubusercontent.com/35541449/235310516-b11d07ac-f3e6-489d-b9bb-a1ad367c0829.png width="500">

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

## Alpha Version
ODM Stats

Version 2 Generated Apr 28, 2023 mAP: 67.9% precision: 76.0%, recall: 67.9%
Training Set: 173 images

ICM Stats

Version 6 Generated Oct 12, 2022 accuracy: 81.3%
Training Set: 2.9k images

## Image Preprocessing
The `preprocess_image` function applies a series of preprocessing steps to the input image before feeding it to the object detection model. Here's a brief explanation of each step:

1. `transforms.Resize(input_size)`: This step resizes the input image so that its smallest dimension becomes equal to the specified `input_size`. The aspect ratio of the image is preserved.

2. `transforms.CenterCrop(input_size)`: This step crops the input image to a square of size `input_size` x `input_size` from the center of the resized image. This is done to obtain a fixed-size input that the model can process.

3. `transforms.ToTensor()`: This step converts the input image from the PIL format (used by the `Image` class) to a PyTorch tensor. This is the expected input format for most PyTorch models. The pixel values are also scaled from the range [0, 255] (integers) to the range [0, 1] (floats).

4. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`: This step normalizes the input tensor using the provided mean and standard deviation values. Normalization is done channel-wise (separately for each color channel) to ensure that each channel has a mean of 0 and a standard deviation of 1. This helps improve the model's performance and training stability.

5. `image = Image.open(image_path).convert("RGB")`: This step opens the image file from the given path and converts it to the RGB format. This is necessary because some image files may have different color modes (e.g., grayscale, RGBA), and the model expects an RGB image as input.

6. `input_tensor = preprocess(image)`: This step applies the composed preprocessing transformations to the input image.

7. `input_batch = input_tensor.unsqueeze(0).to(device)`: This step adds an extra dimension to the input tensor to create a batch with a single image. This is because most deep learning models expect a batch of images as input, even if there's only one image. The tensor is then transferred to the specified device (GPU or CPU) for processing.

## License

This project is released under the MIT License. See the [LICENSE](https://opensource.org/license/mit/) file for more information.
