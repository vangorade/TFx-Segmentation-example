import tensorflow.keras.backend as K

################################################################################
# Helper Functions
################################################################################
def average(x, class_weights=None):
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)

def gather_channels(*xs):
    return xs

def round_if_needed(x, threshold):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x

################################################################################
# Metric Functions
################################################################################
def iou_score(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None):    
    # y_true = K.one_hot(K.squeeze(K.cast(y_true, tf.int32), axis=-1), n_classes)

    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true + y_pred, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, class_weights)

    return score

def dice_coefficient(y_true, y_pred, beta=1.0, class_weights=1., smooth=1e-5, threshold=None):
    # print(y_pred)
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    score = ((1.0 + beta) * tp + smooth) / ((1.0 + beta) * tp + (beta ** 2.0) * fn + fp + smooth)
    # print("Score, wo avg: " + str(score))
    score = average(score, class_weights)
    # print("Score: " + str(score))

    return score

def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    loss = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))

    return K.mean(loss)
