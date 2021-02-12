
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from typing import Union, List, Text
import random

import numpy as np
import albumentations as A
import tensorflow.keras.backend as K

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_advanced_segmentation_models as tasm

import constants


def _transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + "_xf"


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_image_fn(model):
  """Returns a function that feeds the input tensor into the model."""

  @tf.function
  def serve_image_fn(image_tensor):
    """Returns the output to be used in the serving signature.
    
    Args:
        image_tensor: A tensor represeting input image. The image should have 3
                      channels.
    
    Returns:
        The model's predicton on input image tensor
    """
    return model(image_tensor)

  return serve_image_fn

# TFx transform comp will call this func
def preprocessing_fn(inputs: tf.Tensor) -> tf.Tensor:
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      outputs: map from feature keys to raw not-yet-transformed features
    """
    outputs = {}
    
    image_features = tf.map_fn(
        lambda x: tf.io.decode_png(x[0], channels=3),
        inputs[constants.IMAGE_KEY],
        dtype=tf.uint8
        )
    
    image_features = tf.image.resize(image_features, [constants.HEIGHT, constants.WIDTH])
    image_features = tf.image.per_image_standardization(image_features)

    mask_features = tf.map_fn(
        lambda x: tf.io.decode_png(x[0], channels=3),
        inputs[constants.MASK_KEY],
        dtype=tf.uint8
        )

    mask_features = tf.image.resize(mask_features, [constants.HEIGHT, constants.WIDTH])
    mask_features = tf.math.reduce_max(mask_features, axis=-1)
    mask_features = tf.cast(mask_features, dtype=tf.int32)
    mask_features = tf.one_hot(indices=mask_features, depth=constants.N_TOTAL_CLASSES)
    class_values = [constants.TOTAL_CLASSES.index(cls.lower()) for cls in constants.MODEL_CLASSES]
    if not constants.ALL_CLASSES:
        fg_list = []
        bg_list = []
        for mask_num in range(constants.N_TOTAL_CLASSES):
            if mask_num in class_values:
                # Add mask of a class to new_mask
                fg_list.append(mask_features[:, :, :, mask_num])
            else:
                # add all class masks belonging to the background to the background class of the new_mask
                bg_list.append(mask_features[:, :, :, mask_num])

        bg = tf.math.reduce_sum(tf.stack(bg_list, axis=-1), axis=-1, keepdims=False)
        fg_list.append(bg)
        mask_features = tf.stack(fg_list, axis=-1)

    outputs[_transformed_name(constants.IMAGE_KEY)] = image_features
    outputs[_transformed_name(constants.MASK_KEY)] = mask_features
    return outputs
