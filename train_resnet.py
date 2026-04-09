import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch import nn, optim
from PIL import Image
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


class StopSignDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.files = os.listdir(IMAGE_DIR)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_name = self.files[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)

        xml_name = img_name.replace(".png", ".xml")
        xml_path = os.path.join(ANNOTATION_DIR, xml_name)

        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        label = get_label(xml_path)

        return image, torch.tensor(label).long()


dataset = StopSignDataset()
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = resnet18(weights="DEFAULT")

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for imgs, labels in loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.3f}")

torch.save(model.state_dict(), "resnet_stop_model.pth")

print("Training complete.")