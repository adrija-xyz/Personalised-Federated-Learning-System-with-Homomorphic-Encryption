import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from client import FlowerClient, CNN
import flwr as fl
from flwr.server.strategy import FedAvg

# Load data
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# IID partitioning with varying sizes
num_clients = 100
total_samples = len(mnist_train)
labels = np.array(mnist_train.targets)
indices = np.arange(total_samples)
np.random.shuffle(indices)

# Use Dirichlet distribution to vary sizes but keep class distribution similar
proportions = np.random.dirichlet(np.ones(num_clients), size=1)[0]
client_sizes = (proportions * total_samples).astype(int)
diff = total_samples - np.sum(client_sizes)
client_sizes[np.argmax(client_sizes)] += diff

# Create Subset datasets
client_indices = {}
start = 0
for i in range(num_clients):
    end = start + client_sizes[i]
    client_indices[i] = indices[start:end].tolist()
    start = end

client_datasets = [Subset(mnist_train, client_indices[i]) for i in range(num_clients)]

# Define function to start a single client
def client_fn(cid: str):
    cid = int(cid)
    model = CNN()
    train_loader = DataLoader(client_datasets[cid], batch_size=32, shuffle=True)
    numpy_client = FlowerClient(model, train_loader)
    return numpy_client.to_client()

if __name__ == "__main__":
    # Define strategy here before using it
    strategy = FedAvg(
        fraction_fit=0.2,   # Sample 20% of clients each round
        min_fit_clients=20, # Minimum 20 clients per round
        min_available_clients=100
    )
    
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        client_resources={"num_cpus": 1},
        strategy=strategy
    )
    print(hist)