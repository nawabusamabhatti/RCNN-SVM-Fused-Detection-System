import cv2
import joblib
import numpy as np
from skimage import feature
from torchvision import models, transforms
import torch
from xml.dom import minidom

classifier = joblib.load('object_detection_model.pkl')

cnn_model = models.vgg16(pretrained=True)
cnn_model = torch.nn.Sequential(*list(cnn_model.features.children())[:-1])

def extract_lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp_image = feature.local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

def extract_cnn_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = transforms.ToTensor()(image)
    with torch.no_grad():
        cnn_features = model(image.unsqueeze(0))

    return cnn_features

def visualize_detected_objects(image_path):
    image = cv2.imread(image_path)
    lbp_features = extract_lbp_features(image)
    cnn_features = extract_cnn_features(image, cnn_model)
    combined_features = np.concatenate([lbp_features, cnn_features.view(-1).cpu().numpy()])
    label = classifier.predict([combined_features])[0]
    image_with_detection = image.copy()
    cv2.putText(image_with_detection, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(image_with_detection, (0, 0), (image.shape[1], image.shape[0]), (0, 255, 0), 2)
    cv2.imshow('Detected Object', image_with_detection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'Test.jpg'
visualize_detected_objects(image_path)
