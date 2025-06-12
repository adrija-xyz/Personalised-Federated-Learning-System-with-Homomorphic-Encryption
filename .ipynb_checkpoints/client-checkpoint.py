import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from flwr.common import NDArrays, Scalar

# Simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5)
        self.fc = nn.Linear(1440, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 1440)
        return self.fc(x)

# Flower-compatible NumPy client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for batch in self.train_loader:
                x, y = batch
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        for x, y in self.train_loader:
            output = self.model(x)
            loss += self.criterion(output, y).item()
            correct += (output.argmax(1) == y).sum().item()
        return float(loss), len(self.train_loader.dataset), {"accuracy": correct / len(self.train_loader.dataset)}