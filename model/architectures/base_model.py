import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base model class with common utilities
    """

    def __init__(self):
        super(BaseModel, self).__init__()
    
    def count_parameters(self):
        """Count trainable parameters"""
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self):
        """Get model device"""
        
        return next(self.parameters()).device
    
    def save_checkpoint(self, path, epoch, optimizer, loss):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch' : epoch,
            'model_state_dict' : self.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : loss
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path, optimizer=None):
        """Load model checkpoint"""
        
        checkpoint = torch.load(path)

        self.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {path}")
        print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

        return {
            'epoch' : checkpoint['epoch'],
            'loss' : checkpoint['loss']
        }