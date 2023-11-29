import numpy as np


class LSTM:
    """LSTM model implementation.
    Based on Andrej Karpathy's blog post related to RNNs: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    """
    def __init__(self, vocab_size, hidden_size=100, seq_length=25, learning_rate=1e-1):
        # Model parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # LSTM parameters
        self.Wf = np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01  # Forget gate
        self.Wi = np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01  # Input gate
        self.Wc = np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01  # Candidate cell
        self.Wo = np.random.randn(hidden_size, hidden_size + vocab_size) * 0.01  # Output gate
        self.Wy = np.random.randn(vocab_size, hidden_size) * 0.01  # Hidden to output

        self.bf = np.zeros((hidden_size, 1))  # Forget bias
        self.bi = np.zeros((hidden_size, 1))  # Input bias
        self.bc = np.zeros((hidden_size, 1))  # Candidate cell bias
        self.bo = np.zeros((hidden_size, 1))  # Output bias
        self.by = np.zeros((vocab_size, 1))   # Output bias

        # Memory variables for Adagrad
        self.mWf, self.mWi, self.mWc, self.mWo, self.mWy = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo), np.zeros_like(self.Wy)
        self.mbf, self.mbi, self.mbc, self.mbo, self.mby = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo), np.zeros_like(self.by)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def lstm_forward(self, inputs, targets, h_prev, c_prev):
        xs, hs, cs, ys, ps = {}, {}, {}, {}, {}
        hs[-1], cs[-1] = np.copy(h_prev), np.copy(c_prev)
        loss = 0

        # Forward pass
        print(f"Forward pass with {len(inputs)} inputs")
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            z = np.row_stack((hs[t - 1], xs[t]))
            f = self.sigmoid(np.dot(self.Wf, z) + self.bf)
            i = self.sigmoid(np.dot(self.Wi, z) + self.bi)
            c_bar = np.tanh(np.dot(self.Wc, z) + self.bc)
            cs[t] = f * cs[t - 1] + i * c_bar
            o = self.sigmoid(np.dot(self.Wo, z) + self.bo)
            hs[t] = o * np.tanh(cs[t])
            ys[t] = np.dot(self.Wy, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t], 0])


        # Backward pass: gradients
        print(f"Backward pass with {len(inputs)} inputs")
        dWf, dWi, dWc, dWo, dWy = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo), np.zeros_like(self.Wy)
        dbf, dbi, dbc, dbo, dby = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo), np.zeros_like(self.by)
        dh_next, dc_next = np.zeros_like(hs[0]), np.zeros_like(cs[0])

        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Wy.T, dy) + dh_next
            do = dh * np.tanh(cs[t])
            dWo += np.dot(do, np.row_stack((hs[t - 1], xs[t])).T)

            dbo += do
            dc = dc_next + (dh * o * (1 - np.tanh(cs[t]) ** 2))
            dc_bar = dc * i
            dWc += np.dot(dc_bar, np.row_stack((hs[t - 1], xs[t])).T)
            dbc += dc_bar
            di = dc * c_bar
            dWi += np.dot(di, np.row_stack((hs[t - 1], xs[t])).T)
            dbi += di
            df = dc * cs[t - 1]
            dWf += np.dot(df, np.row_stack((hs[t - 1], xs[t])).T)
            dbf += df
            dz = (np.dot(self.Wf.T, df)
                  + np.dot(self.Wi.T, di)
                  + np.dot(self.Wc.T, dc_bar)
                  + np.dot(self.Wo.T, do))
            dh_next = dz[:self.hidden_size, :]
            dc_next = f * dc

        print(f"Clipping gradients")
        for dparam in [dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        print(f"Returning loss and gradients")
        return loss, dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby, hs[len(inputs) - 1], cs[len(inputs) - 1]

    def sample(self, h, c, seed_ix, n):
        """
        Sample a sequence of integers from the model.
        h and c are memory state and cell state, respectively. seed_ix is seed letter for first time step.
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            z = np.row_stack((h, x))
            f = self.sigmoid(np.dot(self.Wf, z) + self.bf)
            i = self.sigmoid(np.dot(self.Wi, z) + self.bi)
            c_bar = np.tanh(np.dot(self.Wc, z) + self.bc)
            c = f * c + i * c_bar
            o = self.sigmoid(np.dot(self.Wo, z) + self.bo)
            h = o * np.tanh(c)
            y = np.dot(self.Wy, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def train(self, data, char_to_ix, ix_to_char):
        n, p = 0, 0
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length

        while True:
            # Prepare inputs
            if p + self.seq_length + 1 >= len(data) or n == 0:
                h_prev = np.zeros((self.hidden_size, 1))
                c_prev = np.zeros((self.hidden_size, 1))
                p = 0

            inputs = [char_to_ix[ch] for ch in data[p:p + self.seq_length]]
            targets = [char_to_ix[ch] for ch in data[p + 1:p + self.seq_length + 1]]

            # Sample from the model now and then
            if n % 100 == 0:
                sample_ix = self.sample(h_prev, c_prev, inputs[0], 200)
                txt = ''.join(ix_to_char[ix] for ix in sample_ix)
                print(f'----\n {txt} \n----')

            # Forward seq_length characters through the net and fetch gradient
            loss, dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby, h_prev, c_prev = self.lstm_forward(inputs, targets, h_prev, c_prev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            if n % 100 == 0:
                print(f'iter {n}, loss: {smooth_loss}')

            # Perform parameter update with Adagrad
            for param, dparam, mem in zip([self.Wf, self.Wi, self.Wc, self.Wo, self.Wy,
                                           self.bf, self.bi, self.bc, self.bo, self.by],
                                          [dWf, dWi, dWc, dWo, dWy,
                                           dbf, dbi, dbc, dbo, dby],
                                          [self.mWf, self.mWi, self.mWc, self.mWo, self.mWy,
                                           self.mbf, self.mbi, self.mbc, self.mbo, self.mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)

            p += self.seq_length  # Move data pointer
            n += 1  # Iteration counter

            # TODO: Implement early stopping?
