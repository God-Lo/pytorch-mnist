import torch
from torch import nn, load
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import NumberRecognition

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

test = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())
dataset = DataLoader(test, 32)

model = NumberRecognition().to(device)

loss_fn = nn.CrossEntropyLoss() 

if __name__ == "__main__":
    with open('model_state.pt', 'rb') as f:
        model.load_state_dict(load(f)) 
    for batch in dataset:
        X,y = batch
        X,y = X.to(device), y.to(device)
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
    print(f"Loss: {loss.item()}")