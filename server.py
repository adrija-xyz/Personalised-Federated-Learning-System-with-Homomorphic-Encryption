import flwr as fl

fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=5)  # Run 5 FL rounds
)