
import tensorflow as tf

def create_base_model(name="ResNet50", weights="imagenet", height=None, width=None,
                      include_top=False, pooling=None, alpha=1.0, depth_multiplier=1.0):
                    # , dropout=0.001):
    if not isinstance(height, int) or not isinstance(width, int):
        raise TypeError("'height' and 'width' need to be of type 'int'")
        
    if name.lower() == "efficientnetb3":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.EfficientNetB3(include_top=include_top, weights=weights, input_shape=[height, width, 3], pooling=pooling)
        layer_names = ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
    elif name.lower() == "resnet50":
        if height <= 31 or width <= 31:
            raise ValueError("Parameters 'height' and 'width' should not be smaller than 32.")
        base_model = tf.keras.applications.ResNet50(include_top=include_top, weights=weights, input_shape=[height, width, 3], pooling=pooling)
        layer_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    else:
        raise ValueError("'name' should be one of, 'efficientnetb3','resnet50'.")
        
    layers = [base_model.get_layer(layer_name).output for layer_name in layer_names]
    

    return base_model, layers, layer_names
