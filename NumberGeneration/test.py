import torch
from torch import load
from model import NumberGeneration
import matplotlib.pyplot as plt 

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NumberGeneration().to(device)

if __name__ == "__main__": 
    with open('model_state.pt', 'rb') as f: 
        model.load_state_dict(load(f))
    x = torch.tensor(int(input("Enter a number: "))).to(device)
    img = model(x).cpu().detach().squeeze().numpy().reshape(28, 28)
    plt.imshow(img, cmap="Greys")
    plt.show()