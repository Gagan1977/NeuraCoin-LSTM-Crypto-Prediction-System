import os
import torch


def get_device():
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

def save_model(model, path):
    
    directory = os.path.dirname(path)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device=None):
    
    if device is None:
        device = get_device()

    state_dict = torch.load(path, map_location=device)
    
    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    print(f"Model loaded from {path}")
    return model


