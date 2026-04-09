import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_score, recall_score, f1_score

IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"

template = cv2.imread("template.png",0)

def get_label(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        if obj.find("name").text.lower() == "stop":
            return 1

    return 0


def multiscale_template_match(image, template):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    best_score = 0

    for scale in np.linspace(0.3,1.0,10):

        resized = cv2.resize(template,None,fx=scale,fy=scale)

        if resized.shape[0] > gray.shape[0] or resized.shape[1] > gray.shape[1]:
            continue

        result = cv2.matchTemplate(gray,resized,cv2.TM_CCOEFF_NORMED)
        _,max_val,_,_ = cv2.minMaxLoc(result)

        best_score = max(best_score,max_val)

    return best_score > 0.4


def apply_blur(image,strength):

    if strength == 0:
        return image

    k = 2*strength + 1
    return cv2.GaussianBlur(image,(k,k),strength)


def evaluate_template(blur_strength):

    y_true = []
    y_pred = []

    for img_name in os.listdir(IMAGE_DIR):

        img_path = os.path.join(IMAGE_DIR,img_name)
        xml_path = os.path.join(ANNOTATION_DIR,img_name.replace(".png",".xml"))

        img = cv2.imread(img_path)

        img = apply_blur(img,blur_strength)

        pred = multiscale_template_match(img,template)

        label = get_label(xml_path)

        y_true.append(label)
        y_pred.append(int(pred))

    precision = precision_score(y_true,y_pred)
    recall = recall_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)

    return precision,recall,f1


results = {}

for blur in [0,1,2,3,4,5,6,7,8,9,10]:

    p,r,f = evaluate_template(blur)

    results[blur] = {
        "precision":p,
        "recall":r,
        "f1":f
    }

with open("results/template_results.json","w") as f:
    json.dump(results,f,indent=4)

print(results)