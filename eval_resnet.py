import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch import nn
from PIL import Image, ImageFilter
from sklearn.metrics import precision_score, recall_score, f1_score
import xml.etree.ElementTree as ET

IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def get_label(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        if obj.find("name").text.lower() == "stop":
            return 1

    return 0


model = resnet18()
model.fc = nn.Linear(model.fc.in_features,2)
model.load_state_dict(torch.load("resnet_stop_model.pth"))
model.to(device)
model.eval()


def evaluate(blur_strength):

    y_true = []
    y_pred = []

    for img_name in os.listdir(IMAGE_DIR):

        img_path = os.path.join(IMAGE_DIR,img_name)
        xml_path = os.path.join(ANNOTATION_DIR,img_name.replace(".png",".xml"))

        img = Image.open(img_path).convert("RGB")

        if blur_strength > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_strength))

        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output).item()

        label = get_label(xml_path)

        y_true.append(label)
        y_pred.append(pred)

    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)

    return precision,recall,f1


results = {}

for blur in [0,1,2,3,4,5,6,7,8,9,10]:

    p,r,f = evaluate(blur)

    results[blur] = {
        "precision":p,
        "recall":r,
        "f1":f
    }

with open("results/resnet_results.json","w") as f:
    json.dump(results,f,indent=4)

print(results)