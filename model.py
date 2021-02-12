import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from typing import Union, List, Text
import random

import numpy as np
import albumentations as A
import tensorflow.keras.backend as K

import tensorflow as tf
import tensorflow_transform as tft
from backbone import create_base_model
from DANet import DANet
from metrics import IOUScore
from losses import CategoricalFocalLoss
from losses import DiceLoss
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


def _image_and_mask_augmentation(image_features, mask_features):
    """Perform image augmentation on batches of images .
    
    Args:
        image_features: a batch of image features
        mask_features: a batch of mask features
    
    Returns:
        The augmented image and mask features
    """
    batch_size = tf.shape(image_features)[0]
    seed = random.random()
    image_features = tf.image.random_flip_left_right(image_features, seed=seed)
    mask_features = tf.image.random_flip_left_right(mask_features, seed=seed)
    image_features = tf.image.resize_with_crop_or_pad(image_features, constants.HEIGHT + 30, constants.WIDTH + 30)
    mask_features = tf.image.resize_with_crop_or_pad(mask_features, constants.HEIGHT + 30, constants.WIDTH + 30)
    image_features = tf.image.random_crop(image_features, (batch_size, constants.HEIGHT, constants.WIDTH, 3), seed=seed)
    mask_features = tf.image.random_crop(mask_features, (batch_size, constants.HEIGHT, constants.WIDTH, len(constants.MODEL_CLASSES) + 1), seed=seed)
    return image_features, mask_features


def _data_augmentation(feature_dict, mask_features):
    """Perform data augmentation on batches of data.
    
    Args:
        feature_dict: a dict containing features of samples
    
    Returns:
        The feature dict with augmented features
    """
    image_features = feature_dict[_transformed_name(constants.IMAGE_KEY)]
    image_features, mask_features = _image_and_mask_augmentation(image_features, mask_features)
    feature_dict[_transformed_name(constants.IMAGE_KEY)] = image_features
    return feature_dict, mask_features


def _input_fn(file_pattern: List[Text], 
              tf_transform_output: tft.TFTransformOutput, 
              batch_size=4,
              is_train: bool=False) -> tf.data.Dataset:
    """Generates features and label for tuning/training.
    
    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of returned
                    dataset to combine in a single batch
        is_train: Whether the input dataset is train split or not

    Returns:
        A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
        )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=_transformed_name(constants.MASK_KEY),
        prefetch_buffer_size=2,
        num_epochs=constants.EPOCHS
        )

    if is_train:
        dataset = dataset.map(lambda x, y: (_data_augmentation(x, y)))

    return dataset

# model 

def get_model(fn_args):
    """
    This function defines a Keras model and returns the model as a Keras object.
    """
    base_model, layers, layer_names = create_base_model(
        name=constants.BACKBONE_NAME,
        weights=constants.PRETRAINED_WEIGHTS,
        height=constants.HEIGHT,
        width=constants.WIDTH,
        include_top=False,
        pooling=None
    )

    model = DANet(
        n_classes=constants.N_MODEL_CLASSES,
        base_model=base_model,
        output_layers=layers,
        backbone_trainable=constants.BACKBONE_TRAINABLE,
        height=constants.HEIGHT,
        width=constants.WIDTH
    ).model()

    opt = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9)
    metrics = [IOUScore(threshold=0.5)]
    categorical_focal_dice_loss = CategoricalFocalLoss(alpha=0.25, gamma=2.0) + DiceLoss()

    model.compile(
        optimizer=opt,
        loss=categorical_focal_dice_loss,
        metrics=metrics,
    )

    return model


# TFX Trainer will call this function.
def run_fn(fn_args):
    """Train the model based on given args.

    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    TrainSetwoAug = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        constants.TRAIN_BATCH_SIZE,
        is_train=True
    )

    ValidationSet = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        constants.EVAL_BATCH_SIZE,
        is_train=False
    )

    model = get_model(fn_args)

    try:
        log_dir = fn_args.model_run_dir
    except KeyError:
        log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    callbacks = [
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="iou_score", factor=0.2, patience=6, verbose=1, mode="max"),
                 tf.keras.callbacks.EarlyStopping(monitor="iou_score", patience=16, mode="max", verbose=1, restore_best_weights=True),
                 tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="batch")
    ]

    print("Start Training")

    model.fit(
        TrainSetwoAug,
        epochs=constants.EPOCHS,
        steps_per_epoch=constants.TRAIN_STEPS,
        validation_data=ValidationSet,
        validation_steps=constants.EVAL_STEPS,
        callbacks=callbacks,
    )

    print("Training Finished")

    signatures = {
        'serving_default':
            _get_serve_image_fn(model).get_concrete_function(
                tf.TensorSpec(
                    shape=[None, constants.HEIGHT, constants.WIDTH, 3],
                    dtype=tf.float32,
                    name=_transformed_name(constants.IMAGE_KEY)
                )
            )
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

    print("Model Saved")
