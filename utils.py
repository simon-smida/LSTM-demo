import numpy as np
from PyQt5.QtWidgets import QMessageBox


# Model parameters
global_training_history = {'loss': [], 'accuracy': []}
model = None

def generate_training_data(input_sequence, context_size, extend=False):
    """Generate training data from the input sequence."""
    data = [int(x) for x in input_sequence.split(',')] 
    
    if extend:
        data *= 10
        
    # Check if data length is sufficient for the given context size
    if len(data) < context_size:
        raise ValueError(f"Input sequence too short for the context size of {context_size}. Please provide a longer sequence or reduce the context size.")

    X, y = [], []    
    for i in range(len(data) - context_size):
        X.append(data[i:i+context_size])
        y.append(data[i+context_size])
    
    X = np.array(X).reshape(-1, context_size, 1)
    y = np.array(y).reshape(-1, 1)
    
    return X, y
