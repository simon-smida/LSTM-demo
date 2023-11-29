import tensorflow as tf

# LSTM Model Definition
def create_model(batch_size, sequence_length):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, stateful=True,
                             batch_input_shape=(batch_size, sequence_length, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False, stateful=True),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model