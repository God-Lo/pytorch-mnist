import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import NumberRecognition

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)

load_data = False
learning_rate = 1e-3
epochs = 10
model = NumberRecognition().to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() 

if __name__ == "__main__":
    if load_data:
        with open('model_state.pt', 'rb') as f: 
            model.load_state_dict(load(f)) 
            
    for epoch in range(epochs):
        for batch in dataset: 
            X,y = batch 
            X,y = X.to(device), y.to(device) 
            y_hat = model(X) 
            loss = loss_fn(y_hat, y) 

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
        print(f"Epoch: {epoch}; Loss: {loss.item()}")
    
    with open('model_state.pt', 'wb') as f: 
        save(model.state_dict(), f) 