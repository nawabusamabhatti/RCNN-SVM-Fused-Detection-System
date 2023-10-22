import os
import cv2
import numpy as np
from sklearn.svm import SVC
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import joblib
from xml.dom import minidom
from tqdm import tqdm

image_folder = "Dataset/Images"
annotation_folder = "Dataset/Annotations"

whiteboard_label = "Whiteboard"
nameplate_label = "Nameplate"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def extract_lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp_image = feature.local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    return lbp_hist

cnn_model = models.vgg16(pretrained=True)
cnn_model = nn.Sequential(*list(cnn_model.features.children())[:-1])  
cnn_model = cnn_model.to(device)

features = []
labels = []
cnn_model.eval()
def extract_cnn_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = transforms.ToTensor()(image).to(device)
    image = image.unsqueeze(0)  
    with torch.no_grad():
        cnn_features = model(image)

    return cnn_features

for filename in tqdm(os.listdir(image_folder), unit="image"):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        image = cv2.imread(os.path.join(image_folder, filename))
        label = "Other"
        annotation_file = os.path.join(annotation_folder, os.path.splitext(filename)[0] + ".xml")
        if os.path.isfile(annotation_file):
            xml_doc = minidom.parse(annotation_file)
            objects = xml_doc.getElementsByTagName("object")
            
            if objects:
                label_element = objects[0].getElementsByTagName("name")
                if label_element:
                    label = label_element[0].firstChild.nodeValue
        lbp_features = extract_lbp_features(image)
        cnn_features = extract_cnn_features(image, cnn_model)
        cnn_features = cnn_features.view(-1).cpu().detach().numpy()
        combined_features = np.concatenate([lbp_features, cnn_features])
        features.append(combined_features)
        labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
classifier = SVC()
classifier.fit(X_train, y_train)
joblib.dump(classifier, 'object_detection_model.pkl')
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
