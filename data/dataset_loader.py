import tensorflow as tf

def get_dummy_dataset(seq_len, batch_size):
    # Dummy random int dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.random.uniform([10000, seq_len + 1], maxval=50000, dtype=tf.int32)
    )

    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]

    dataset = dataset.map(split_input_target)
    dataset = dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
