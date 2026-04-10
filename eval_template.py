import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_score, recall_score, f1_score

IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"

if not os.path.exists("results"):
    os.makedirs("results")

template = cv2.imread("template.png",0)

def get_label(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        if obj.find("name").text.lower() == "stop":
            return 1

    return 0


def multiscale_template_match(image, template, threshold):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    best_score = 0

    for scale in np.linspace(0.3,1.0,10):

        resized = cv2.resize(template,None,fx=scale,fy=scale)

        if resized.shape[0] > gray.shape[0] or resized.shape[1] > gray.shape[1]:
            continue

        result = cv2.matchTemplate(gray,resized,cv2.TM_CCOEFF_NORMED)
        _,max_val,_,_ = cv2.minMaxLoc(result)

        best_score = max(best_score,max_val)

    return best_score > threshold


def apply_blur(image,strength):

    if strength == 0:
        return image

    k = 2*strength + 1
    return cv2.GaussianBlur(image,(k,k),strength)


def evaluate_template(blur_strength, threshold):

    y_true = []
    y_pred = []

    for img_name in os.listdir(IMAGE_DIR):

        img_path = os.path.join(IMAGE_DIR,img_name)
        xml_path = os.path.join(ANNOTATION_DIR,img_name.replace(".png",".xml"))

        img = cv2.imread(img_path)

        img = apply_blur(img,blur_strength)

        pred = multiscale_template_match(img,template,threshold)

        label = get_label(xml_path)

        y_true.append(label)
        y_pred.append(int(pred))

    precision = precision_score(y_true,y_pred,zero_division=0)
    recall = recall_score(y_true,y_pred,zero_division=0)
    f1 = f1_score(y_true,y_pred,zero_division=0)

    return precision,recall,f1


for target_threshold in [0.4, 0.8]:
    results = {}
    print(f"Processing evaluation for threshold: {target_threshold}...")

    for blur in [0,1,2,3,4,5,6,7,8,9,10]:

        p,r,f = evaluate_template(blur, target_threshold)

        results[blur] = {
            "precision":p,
            "recall":r,
            "f1":f
        }

    output_filename = f"results/template_results_{target_threshold}.json"
    with open(output_filename,"w") as f:
        json.dump(results,f,indent=4)

print(f"Saved: {output_filename}")