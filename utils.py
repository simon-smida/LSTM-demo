import numpy as np

def generate_training_data(input_sequence, context_size, repeat_sequence=True):
    """
    Generate training data from the input sequence.

    Args:
    input_sequence (str): A comma-separated string of integers.
    context_size (int): The number of elements in each input sequence for the LSTM.
    repeat_sequence (bool): If True, repeats the sequence to enlarge the dataset.

    Returns:
    tuple: Two Numpy arrays, X (input sequences) and y (output values).
    """
    try:
        data = [int(x) for x in input_sequence.split(',')]
    except ValueError:
        raise ValueError("Input sequence must be a comma-separated list of integers.")

    if len(data) < context_size:
        raise ValueError(f"Input sequence length ({len(data)}) is less than the context size ({context_size}).")

    if repeat_sequence:
        # Repeat the sequence to enlarge the dataset. 
        # Note: This is for demonstration purposes and might not be ideal for real training scenarios.
        data *= 10

    X, y = [], []
    for i in range(len(data) - context_size):
        X.append(data[i:i + context_size])
        y.append(data[i + context_size])
    X = np.array(X).reshape(-1, context_size, 1)
    y = np.array(y).reshape(-1, 1)

    return X, y