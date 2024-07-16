import torch
from torchvision import transforms
from PIL import Image

model = torch.jit.load('QATDogCatMobileNetV2_224.pt')
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def modelpred(filepath):
    image = Image.open(filepath)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

        return confidence, predicted_class
