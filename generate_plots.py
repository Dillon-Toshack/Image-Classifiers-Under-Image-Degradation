import os
import json
import matplotlib.pyplot as plt

if not os.path.exists("figures"):
    os.makedirs("figures")

with open("results/template_results_0.4.json") as f:
    temp_04 = json.load(f)

with open("results/template_results_0.8.json") as f:
    temp_08 = json.load(f)

with open("results/resnet_results.json") as f:
    resnet = json.load(f)

blur_levels = [0,1,2,3,4,5,6,7,8,9,10]

t04_f1 = [temp_04[str(b)]["f1"] for b in blur_levels]
t08_f1 = [temp_08[str(b)]["f1"] for b in blur_levels]
resnet_f1 = [resnet[str(b)]["f1"] for b in blur_levels]

t04_prec = [temp_04[str(b)]["precision"] for b in blur_levels]
t08_prec = [temp_08[str(b)]["precision"] for b in blur_levels]
resnet_prec = [resnet[str(b)]["precision"] for b in blur_levels]

t04_rec = [temp_04[str(b)]["recall"] for b in blur_levels]
t08_rec = [temp_08[str(b)]["recall"] for b in blur_levels]
resnet_rec = [resnet[str(b)]["recall"] for b in blur_levels]


plt.figure(figsize=(10, 6))
plt.plot(blur_levels, t04_f1, label="Template (Thresh 0.4)", marker='o')
plt.plot(blur_levels, t08_f1, label="Template (Thresh 0.8)", marker='s')
plt.plot(blur_levels, resnet_f1, label="ResNet", marker='^', linewidth=2)
plt.xlabel("Blur Strength")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Blur")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figures/f1_vs_blur.png")

plt.figure(figsize=(10, 6))
plt.plot(blur_levels, t04_prec, label="Template (Thresh 0.4)", marker='o')
plt.plot(blur_levels, t08_prec, label="Template (Thresh 0.8)", marker='s')
plt.plot(blur_levels, resnet_prec, label="ResNet", marker='^', linewidth=2)
plt.xlabel("Blur Strength")
plt.ylabel("Precision")
plt.title("Precision vs Blur")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figures/precision_vs_blur.png")

plt.figure(figsize=(10, 6))
plt.plot(blur_levels, t04_rec, label="Template (Thresh 0.4)", marker='o')
plt.plot(blur_levels, t08_rec, label="Template (Thresh 0.8)", marker='s')
plt.plot(blur_levels, resnet_rec, label="ResNet", marker='^', linewidth=2)
plt.xlabel("Blur Strength")
plt.ylabel("Recall")
plt.title("Recall vs Blur")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figures/recall_vs_blur.png")

plt.show()