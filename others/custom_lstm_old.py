import numpy as np
import matplotlib.pyplot as plt



class LSTM:
    """
    LSTM (Long Short-Term Memory) network implementation.
    This implementation is inspired by Andrej Karpathy's blog post on RNNs.
    """
    
    def __init__(
        self, vocab_size, hidden_size=100, sequence_length=25, learning_rate=1
    ):
        """
        Initialize the LSTM model with specified parameters.
        """
        # Model parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate

        # Initialize LSTM gate weights and biases
        # self.forget_gate_weights = (
        #     np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01 
        # )
        # self.input_gate_weights = (
        #     np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01
        # )
        # self.cell_candidate_weights = (
        #     np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01
        # )
        # self.output_gate_weights = (
        #     np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01
        # )
        # self.output_weights = np.random.randn(vocab_size, hidden_size) * 0.01
        
        # Xavier/Glorot Initialization
        stddev = np.sqrt(2 / (hidden_size + vocab_size))
        self.forget_gate_weights = np.random.normal(0, stddev, (hidden_size, hidden_size + vocab_size))
        self.input_gate_weights = np.random.normal(0, stddev, (hidden_size, hidden_size + vocab_size))
        self.cell_candidate_weights = np.random.normal(0, stddev, (hidden_size, hidden_size + vocab_size))
        self.output_gate_weights = np.random.normal(0, stddev, (hidden_size, hidden_size + vocab_size))
        self.output_weights = np.random.normal(0, stddev, (vocab_size, hidden_size))
        
        # Bias vectors for gates
        self.forget_bias = np.zeros((hidden_size, 1))
        self.input_bias = np.zeros((hidden_size, 1))
        self.cell_candidate_bias = np.zeros((hidden_size, 1))
        self.output_bias = np.zeros((hidden_size, 1))
        self.output_layer_bias = np.zeros((vocab_size, 1))

        # Initialize memory variables for Adagrad optimization
        self.memory = {
            name: np.zeros_like(value)
            for name, value in self.__dict__.items()
            if "weights" in name or "bias" in name
        }
        
        # Adam optimizer parameters
        self.m = {k: np.zeros_like(v) for k, v in self.__dict__.items() if "weights" in k or "bias" in k}
        self.v = {k: np.zeros_like(v) for k, v in self.__dict__.items() if "weights" in k or "bias" in k}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def lstm_forward(self, inputs, targets, previous_hidden_state, previous_cell_state):
        """
        Forward pass of the LSTM for a sequence of inputs.
        """
        input_vectors, hidden_states, cell_states, outputs, probabilities = (
            {},
            {},
            {},
            {},
            {},
        )
        hidden_states[-1], cell_states[-1] = np.copy(previous_hidden_state), np.copy(
            previous_cell_state
        )
        loss = 0

        # Forward pass through the sequence
        for t in range(len(inputs)):
            input_vectors[t] = np.zeros((self.vocab_size, 1))
            input_vectors[t][inputs[t]] = 1  # One-hot encoding of the input character

            combined_input = np.row_stack(
                (hidden_states[t - 1], input_vectors[t])
            )  # Concatenate previous hidden state with current input
            forget_gate = self.sigmoid(
                np.dot(self.forget_gate_weights, combined_input) + self.forget_bias
            )  # Forget gate
            input_gate = self.sigmoid(
                np.dot(self.input_gate_weights, combined_input) + self.input_bias
            )  # Input gate
            cell_input = np.tanh(
                np.dot(self.cell_candidate_weights, combined_input)
                + self.cell_candidate_bias
            )  # Cell input
            cell_states[t] = (
                forget_gate * cell_states[t - 1] + input_gate * cell_input
            )  # Update cell state
            output_gate = self.sigmoid(
                np.dot(self.output_gate_weights, combined_input) + self.output_bias
            )  # Output gate
            hidden_states[t] = output_gate * np.tanh(
                cell_states[t]
            )  # Update hidden state

            raw_outputs = (
                np.dot(self.output_weights, hidden_states[t]) + self.output_layer_bias
            )  # Compute raw outputs
            probabilities[t] = np.exp(raw_outputs) / np.sum(
                np.exp(raw_outputs)
            )  # Softmax for probability distribution
            loss += -np.log(probabilities[t][targets[t], 0])  # Cross-entropy loss

        # Backward pass: Initialize gradients
        gradients = {
            name: np.zeros_like(value)
            for name, value in self.__dict__.items()
            if "weights" in name or "bias" in name
        }
        next_hidden_gradient, next_cell_gradient = np.zeros_like(
            previous_hidden_state
        ), np.zeros_like(previous_cell_state)

        # Backward pass through the sequence
        for t in reversed(range(len(inputs))):
            dy = np.copy(probabilities[t])
            dy[targets[t]] -= 1  # Derivative of cross-entropy loss
            gradients["output_weights"] += np.dot(dy, hidden_states[t].T)
            gradients["output_layer_bias"] += dy

            dh = np.dot(self.output_weights.T, dy) + next_hidden_gradient
            do = dh * np.tanh(cell_states[t])
            gradients["output_gate_weights"] += np.dot(do, combined_input.T)
            gradients["output_bias"] += do

            dc = next_cell_gradient + (
                dh * output_gate * (1 - np.tanh(cell_states[t]) ** 2)
            )
            dc_input = dc * input_gate
            gradients["cell_candidate_weights"] += np.dot(dc_input, combined_input.T)
            gradients["cell_candidate_bias"] += dc_input

            di = dc * cell_input
            gradients["input_gate_weights"] += np.dot(di, combined_input.T)
            gradients["input_bias"] += di

            df = dc * cell_states[t - 1]
            gradients["forget_gate_weights"] += np.dot(df, combined_input.T)
            gradients["forget_bias"] += df

            d_combined_input = (
                np.dot(self.forget_gate_weights.T, df)
                + np.dot(self.input_gate_weights.T, di)
                + np.dot(self.cell_candidate_weights.T, dc_input)
                + np.dot(self.output_gate_weights.T, do)
            )
            next_hidden_gradient = d_combined_input[: self.hidden_size, :]
            next_cell_gradient = forget_gate * dc

        # Apply gradient clipping
        for gradient in gradients.values():
            np.clip(gradient, -5, 5, out=gradient)

        return (
            loss,
            gradients,
            hidden_states[len(inputs) - 1],
            cell_states[len(inputs) - 1],
        )

    def sample(self, h, c, seed_index, n):
        """
        Sample a sequence of integers from the model.
        h and c are memory state and cell state, respectively. seed_index is the seed letter for the first time step.
        n is the number of characters to sample after the seed character.
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_index] = 1
        indices = []
        
        for t in range(n):
            combined_input = np.row_stack((h, x))
            forget_gate = self.sigmoid(
                np.dot(self.forget_gate_weights, combined_input) + self.forget_bias
            )
            input_gate = self.sigmoid(
                np.dot(self.input_gate_weights, combined_input) + self.input_bias
            )
            cell_input = np.tanh(
                np.dot(self.cell_candidate_weights, combined_input)
                + self.cell_candidate_bias
            )
            c = forget_gate * c + input_gate * cell_input
            output_gate = self.sigmoid(
                np.dot(self.output_gate_weights, combined_input) + self.output_bias
            )
            h = output_gate * np.tanh(c)
            raw_output = np.dot(self.output_weights, h) + self.output_layer_bias
            # Stable Softmax
            raw_output -= np.max(raw_output)  # Subtracting the max value for numerical stability
            p = np.exp(raw_output) / np.sum(np.exp(raw_output))
            
            index = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[index] = 1
            indices.append(index)
        return indices

    def train(self, data, char_to_index, index_to_char, num_epochs=10, lr_decay=0.9):
        """
        Train the LSTM model on the given data using epochs.
        """
        data_size = len(data)
        # Plot loss 
        loss_history = []
            
        for epoch in range(num_epochs):
            pointer = 0
            previous_hidden_state = np.zeros((self.hidden_size, 1))
            previous_cell_state = np.zeros((self.hidden_size, 1))
            smooth_loss = -np.log(1.0 / self.vocab_size) * self.sequence_length

            while pointer + self.sequence_length + 1 < data_size:
                # Prepare inputs and targets
                if pointer + self.sequence_length + 1 >= len(data) or epoch == 0 and pointer == 0:
                    previous_hidden_state = np.zeros((self.hidden_size, 1))  # Reset LSTM memory
                    previous_cell_state = np.zeros((self.hidden_size, 1))  # Reset LSTM memory

                inputs = [char_to_index[char] for char in data[pointer:pointer + self.sequence_length]]
                targets = [char_to_index[char] for char in data[pointer + 1:pointer + self.sequence_length + 1]]

                # Sample from the model now and then
                if epoch % 100 == 0 and pointer == 0:
                    sample_indices = self.sample(previous_hidden_state, previous_cell_state, inputs[0], 200)
                    text_sample = "".join(index_to_char[index] for index in sample_indices)
                    print(f"----\n {text_sample} \n----")

                # Forward and backward passes
                loss, gradients, previous_hidden_state, previous_cell_state = self.lstm_forward(inputs, targets, previous_hidden_state, previous_cell_state)
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
                loss_history.append(smooth_loss)
                

                if epoch % 10 == 0 and pointer == 0:
                    print(f"Epoch {epoch}, Iteration {pointer}, Loss: {smooth_loss}")

                # Perform parameter update with Adagrad
                # for param, dparam, mem in zip(
                #     [self.forget_gate_weights, self.input_gate_weights, self.cell_candidate_weights, self.output_gate_weights, self.output_weights, self.forget_bias, self.input_bias, self.cell_candidate_bias, self.output_bias, self.output_layer_bias],
                #     [gradients["forget_gate_weights"], gradients["input_gate_weights"], gradients["cell_candidate_weights"], gradients["output_gate_weights"], gradients["output_weights"], gradients["forget_bias"], gradients["input_bias"], gradients["cell_candidate_bias"], gradients["output_bias"], gradients["output_layer_bias"]],
                #     [self.memory["forget_gate_weights"], self.memory["input_gate_weights"], self.memory["cell_candidate_weights"], self.memory["output_gate_weights"], self.memory["output_weights"], self.memory["forget_bias"], self.memory["input_bias"], self.memory["cell_candidate_bias"], self.memory["output_bias"], self.memory["output_layer_bias"]],
                # ):
                #     mem += dparam * dparam
                #     param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)
                for param_name in self.m.keys():
                    dparam = gradients[param_name]
                    m = self.m[param_name]
                    v = self.v[param_name]

                    m = self.beta1 * m + (1 - self.beta1) * dparam
                    v = self.beta2 * v + (1 - self.beta2) * (dparam ** 2)

                    m_corr = m / (1 - self.beta1 ** (epoch + 1))
                    v_corr = v / (1 - self.beta2 ** (epoch + 1))

                    self.__dict__[param_name] -= self.learning_rate * m_corr / (np.sqrt(v_corr) + self.epsilon)

                    # Update the optimizer state
                    self.m[param_name] = m
                    self.v[param_name] = v
                    
                pointer += self.sequence_length
                
            # Learning rate decay after each epoch
            self.learning_rate *= lr_decay
            print(f"Epoch {epoch + 1}/{num_epochs} completed")

        print("Training completed, plotting loss history...")
        
        # Plot loss history
        plt.plot(loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()

        return loss_history
        # TODO: Implement early stopping?
            
    
    
            
