import tensorflow as tf
import pathlib
import os
import numpy as np

class ProcessImage:
    def __init__(self, path, img_height, img_width, batch_size):
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.data_dir = pathlib.Path(path)
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def get_files(self, data_dir):
        ds_list = tf.data.Dataset.list_files(str(data_dir/'*'), shuffle=False)
        image_count = len(list(data_dir.glob(f'*.jpg')))
        ds_list = ds_list.shuffle(image_count, reshuffle_each_iteration=False)
        return ds_list

    def decode_img(self, img):
        img = tf.image.decode_jpeg(img, channels=3)
        return tf.image.resize(img, [self.img_height, self.img_width])

    def process_path(self, file_path):
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img

    def process(self):
        ds_list = self.get_files(self.data_dir)
        ds = ds_list.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.cache()
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds