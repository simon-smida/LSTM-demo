import numpy as np


# Model parameters
context_size = 4
global_training_history = {'loss': [], 'accuracy': []}
model = None

def generate_training_data(input_sequence, context_size):
    """Generate training data from the input sequence."""
    data = [int(x) for x in input_sequence.split(',')] 
    # Enlarge the sequence by repeating it
    data *= 10
    X, y = [], []
    for i in range(len(data) - context_size):
        X.append(data[i:i+context_size])
        y.append(data[i+context_size])
    X = np.array(X).reshape(-1, context_size, 1)
    y = np.array(y).reshape(-1, 1)
    
    return X, y
