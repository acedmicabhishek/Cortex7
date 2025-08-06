import tensorflow as tf
from models.cortex_model import CortexModel
from data.dataset_loader import get_dummy_dataset

def train_model(config):
    model = CortexModel(config)
    dataset = get_dummy_dataset(config["seq_len"], config["batch_size"])

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    model.fit(dataset, epochs=config["epochs"])
