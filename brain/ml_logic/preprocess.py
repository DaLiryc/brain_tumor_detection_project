import tensorflow as tf
from brain.params import *


def load_process_image(path, target_size, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)

    return tf.clip_by_value(img, 0.0, 1.0)


def pipeline_building(df, batch_size) :

    paths = df["image_path"].values.astype(str)
    labels = df["tumor_type_encoded"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(len(paths))
    ds = ds.map(
        lambda x, y: (load_process_image(x, TARGET_SIZE ,augment=True), y),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)

    return ds
