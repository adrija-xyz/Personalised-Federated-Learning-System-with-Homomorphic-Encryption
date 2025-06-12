# Save this as app.py and run with: flwr run app.py

from flwr.common import Context

def client_fn(context: Context):
    # Access cid through the properties dictionary
    cid = int(context.properties["cid"])
    model = CNN()
    train_loader = DataLoader(client_datasets[cid], batch_size=32, shuffle=True)
    numpy_client = FlowerClient(model, train_loader)
    return numpy_client.to_client()

# Then run with: flwr run app.py