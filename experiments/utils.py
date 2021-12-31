import torch

def no_grad(function):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return function(*args, **kwargs)
    return wrapper