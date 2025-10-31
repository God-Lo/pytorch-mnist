import torch
from torch import load
from torchvision.transforms import ToTensor
from model import NumberRecognition
from PIL import Image

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NumberRecognition().to(device)

if __name__ == "__main__": 
    with open('model_state.pt', 'rb') as f: 
        model.load_state_dict(load(f))
    img = Image.open('img_1.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    print(torch.argmax(model(img_tensor)))