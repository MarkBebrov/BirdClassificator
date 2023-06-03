from torchvision import transforms
from PIL import Image
import torch
from load_data import load_data

def predict_image(image_path):
    model = torch.load('trained_model.pth')
    model = model.eval()

    image = Image.open(image_path)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0)
    
    _, _, class_names = load_data()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        print(f"Predicted class: {predicted_class}")
        return predicted_class

        

