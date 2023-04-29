from roboflow import Roboflow
rf = Roboflow(api_key="Ph3dNUBinQO7VR7goYsB")
project = rf.workspace().project("nike-adidas-and-converse-shoes-classification")
model = project.version(6).model

# infer on a local image
print(model.predict("IMAGE_FILENAME", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("IMAGE_FILENAME", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())