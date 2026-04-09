import json
import matplotlib.pyplot as plt

with open("results/template_results.json") as f:
    template = json.load(f)

with open("results/resnet_results.json") as f:
    resnet = json.load(f)

blur_levels = [0,1,2,3,4,5,6,7,8,9,10]

template_f1 = [template[str(b)]["f1"] for b in blur_levels]
resnet_f1 = [resnet[str(b)]["f1"] for b in blur_levels]

template_precision = [template[str(b)]["precision"] for b in blur_levels]
resnet_precision = [resnet[str(b)]["precision"] for b in blur_levels]

template_recall = [template[str(b)]["recall"] for b in blur_levels]
resnet_recall = [resnet[str(b)]["recall"] for b in blur_levels]


plt.figure()
plt.plot(blur_levels,template_f1,label="Template Matching")
plt.plot(blur_levels,resnet_f1,label="ResNet")
plt.xlabel("Blur Strength")
plt.ylabel("F1 Score")
plt.title("F1 vs Blur")
plt.legend()
plt.savefig("figures/f1_vs_blur.png")


plt.figure()
plt.plot(blur_levels,template_precision,label="Template Matching")
plt.plot(blur_levels,resnet_precision,label="ResNet")
plt.xlabel("Blur Strength")
plt.ylabel("Precision")
plt.title("Precision vs Blur")
plt.legend()
plt.savefig("figures/precision_vs_blur.png")


plt.figure()
plt.plot(blur_levels,template_recall,label="Template Matching")
plt.plot(blur_levels,resnet_recall,label="ResNet")
plt.xlabel("Blur Strength")
plt.ylabel("Recall")
plt.title("Recall vs Blur")
plt.legend()
plt.savefig("figures/recall_vs_blur.png")

plt.show()