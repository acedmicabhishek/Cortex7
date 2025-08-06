import tensorflow as tf
from models.cortex_block import CortexBlock

class CortexModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(config["vocab_size"], config["embedding_dim"])
        self.blocks = [CortexBlock(config) for _ in range(config["num_layers"])]
        self.final_ln = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(config["vocab_size"])

    def call(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.head(x)
