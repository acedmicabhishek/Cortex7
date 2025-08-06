import tensorflow as tf

class CortexBlock(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = tf.keras.layers.Dense(config["embedding_dim"], activation='relu')

    def call(self, x):
        return self.dense(x)
