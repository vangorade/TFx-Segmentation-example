
import tensorflow as tf
import tensorflow.keras.backend as K
class ConvolutionBnActivation(tf.keras.layers.Layer):
    """
    """
    # def __init__(self, filters, kernel_size, strides=(1, 1), activation=tf.keras.activations.relu, **kwargs):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", data_format=None, dilation_rate=(1, 1),
                 groups=1, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_batchnorm=False, 
                 axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, trainable=True,
                 post_activation="relu", block_name=None):
        super(ConvolutionBnActivation, self).__init__()


        # 2D Convolution Arguments
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = not (use_batchnorm)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Batch Normalization Arguments
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.trainable = trainable
        
        self.block_name = block_name
        
        self.conv = None
        self.bn = None
        #tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.post_activation = tf.keras.layers.Activation(post_activation)

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=self.block_name + "_conv" if self.block_name is not None else None

        )

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            trainable=self.trainable,
            name=self.block_name + "_bn" if self.block_name is not None else None
        )

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.post_activation(x)

        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1], input_shape[2], self.filters]

    
class FPNBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(FPNBlock, self).__init__()

        self.filters = filters
        self.input_filters = None

        self.conv1x1_1 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_uniform")
        self.conv1x1_2 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", kernel_initializer="he_uniform")

        self.upsample2d = tf.keras.layers.UpSampling2D((2, 2))
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        if input_shape != self.filters:
            self.input_filters = True

    def call(self, x, skip, training=None):
        if self.input_filters:
            x = self.conv1x1_1(x)

        skip = self.conv1x1_2(skip)
        x = self.upsample2d(x)
        x = self.add([x, skip])

        return x
    
class PAM_Module(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(PAM_Module, self).__init__()

        self.filters = filters

        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.concat = tf.keras.layers.Concatenate(axis=axis)

        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, (1, 1))
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(filters, (1, 1))
        self.conv1x1_bn_relu_3 = ConvolutionBnActivation(filters, (1, 1))

        self.gamma = None

        self.softmax = tf.keras.layers.Activation("softmax")
    
    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1,),
            initializer="random_normal",
            name="pam_gamma",
            trainable=True,
            )
    
    def call(self, x, training=None):
        BS, C, H, W = x.shape

        query = self.conv1x1_bn_relu_1(x, training=training)
        key = self.conv1x1_bn_relu_2(x, training=training)

        if K.image_data_format() == "channels_last":
            query = tf.keras.layers.Reshape((H * W, -1))(query) 
            key = tf.keras.layers.Reshape((H * W, -1))(key)

            energy = tf.linalg.matmul(query, key, transpose_b=True)
        else:
            query = tf.keras.layers.Reshape((-1, H * W))(query)
            key = tf.keras.layers.Reshape((-1, H * W))(key)
            
            energy = tf.linalg.matmul(query, key, transpose_a=True)
        
        attention = self.softmax(energy)

        value = self.conv1x1_bn_relu_3(x, training=training)

        if K.image_data_format() == "channels_last":
            value = tf.keras.layers.Reshape((H * W, -1))(value) 
            out = tf.linalg.matmul(value, attention, transpose_a=True)
        else:
            value = tf.keras.layers.Reshape((-1, H * W))(value)
            out = tf.linalg.matmul(value, attention)

        out = tf.keras.layers.Reshape(x.shape[1:])(out)
        out = self.gamma * out + x

        return out

class CAM_Module(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(CAM_Module, self).__init__()

        self.filters = filters

        self.gamma = None

        self.softmax = tf.keras.layers.Activation("softmax")

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1,),
            initializer="random_normal",
            name="cam_gamma",
            trainable=True,
            )
    
    def call(self, x, training=None):
        BS, C, H, W = x.shape

        if K.image_data_format() == "channels_last":
            query = tf.keras.layers.Reshape((-1, C))(x) 
            key = tf.keras.layers.Reshape((-1, C))(x)

            energy = tf.linalg.matmul(query, key, transpose_a=True)
            energy_2 = tf.math.reduce_max(energy, axis=1, keepdims=True)[0] - energy
        else:
            query = tf.keras.layers.Reshape((C, -1))(query)
            key = tf.keras.layers.Reshape((C, -1))(key)
        
            energy = tf.linalg.matmul(query, key, transpose_b=True)
            energy_2 = tf.math.reduce_max(energy, axis=-1, keepdims=True)[0] - energy
        
        attention = self.softmax(energy_2)

        if K.image_data_format() == "channels_last":
            value = tf.keras.layers.Reshape((-1, C))(x)
            out = tf.linalg.matmul(attention, value, transpose_b=True) 
        else:
            value = tf.keras.layers.Reshape((C, -1))(x)
            out = tf.linalg.matmul(attention, value)

        out = tf.keras.layers.Reshape(x.shape[1:])(out)
        out = self.gamma * out + x

        return out

