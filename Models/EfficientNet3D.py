from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import InputSpec
import tensorflow.keras.utils as conv_utils
import six
import warnings
from distutils.version import StrictVersion
from six.moves import xrange
import string
import tensorflow as tf
import math

import collections



# other codes https://github.com/ZFTurbo/efficientnet_3D based 



def generate_legacy_interface(allowed_positional_args=None,
                              conversions=None,
                              preprocessor=None,
                              value_conversions=None):
    allowed_positional_args = allowed_positional_args or []
    conversions = conversions or []
    value_conversions = value_conversions or []

    def legacy_support(func):
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            layer_name = args[0].__class__.__name__
            if preprocessor:
                args, kwargs, converted = preprocessor(args, kwargs)
            else:
                converted = []
            if len(args) > len(allowed_positional_args) + 1:
                raise TypeError('Layer `' + layer_name +
                                '` can accept only ' +
                                str(len(allowed_positional_args)) +
                                ' positional arguments (' +
                                str(allowed_positional_args) + '), but '
                                'you passed the following '
                                'positional arguments: ' +
                                str(args[1:]))
            for key in value_conversions:
                if key in kwargs:
                    old_value = kwargs[key]
                    if old_value in value_conversions[key]:
                        kwargs[key] = value_conversions[key][old_value]
            for old_name, new_name in conversions:
                if old_name in kwargs:
                    value = kwargs.pop(old_name)
                    kwargs[new_name] = value
                    converted.append((new_name, old_name))
            if converted:
                signature = '`' + layer_name + '('
                for value in args[1:]:
                    if isinstance(value, six.string_types):
                        signature += '"' + value + '"'
                    else:
                        signature += str(value)
                    signature += ', '
                for i, (name, value) in enumerate(kwargs.items()):
                    signature += name + '='
                    if isinstance(value, six.string_types):
                        signature += '"' + value + '"'
                    else:
                        signature += str(value)
                    if i < len(kwargs) - 1:
                        signature += ', '
                signature += ')`'
                warnings.warn('Update your `' + layer_name +
                              '` layer call to the Keras 2 API: ' + signature)
            return func(*args, **kwargs)
        return wrapper
    return legacy_support


def conv3d_args_preprocessor(args, kwargs):
    if len(args) > 5:
        raise TypeError('Layer can receive at most 4 positional arguments.')
    if len(args) == 5:
        if isinstance(args[2], int) and isinstance(args[3], int) and isinstance(args[4], int):
            kernel_size = (args[2], args[3], args[4])
            args = [args[0], args[1], kernel_size]
    elif len(args) == 4 and isinstance(args[3], int):
        if isinstance(args[2], int) and isinstance(args[3], int):
            new_keywords = ['padding', 'strides', 'data_format']
            for kwd in new_keywords:
                if kwd in kwargs:
                    raise ValueError(
                        'It seems that you are using the Keras 2 '
                        'and you are passing both `kernel_size` and `strides` '
                        'as integer positional arguments. For safety reasons, '
                        'this is disallowed. Pass `strides` '
                        'as a keyword argument instead.')
        if 'kernel_dim3' in kwargs:
            kernel_size = (args[2], args[3], kwargs.pop('kernel_dim3'))
            args = [args[0], args[1], kernel_size]
    elif len(args) == 3:
        if 'kernel_dim2' in kwargs and 'kernel_dim3' in kwargs:
            kernel_size = (args[2],
                            kwargs.pop('kernel_dim2'),
                            kwargs.pop('kernel_dim3'))
            args = [args[0], args[1], kernel_size]
    elif len(args) == 2:
        if 'kernel_dim1' in kwargs and 'kernel_dim2' in kwargs and 'kernel_dim3' in kwargs:
            kernel_size = (kwargs.pop('kernel_dim1'),
                            kwargs.pop('kernel_dim2'),
                            kwargs.pop('kernel_dim3'))
            args = [args[0], args[1], kernel_size]
    return args, kwargs, [('kernel_size', 'kernel_dim*')]


def _preprocess_padding(padding):
    """Convert keras' padding to tensorflow's padding.
    # Arguments
        padding: string, `"same"` or `"valid"`.
    # Returns
        a string, `"SAME"` or `"VALID"`.
    # Raises
        ValueError: if `padding` is invalid.
    """
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid padding: ' + str(padding))
    return padding


def dtype(x):
    return x.dtype.base_dtype.name


def _has_nchw_support():
    return True


def _preprocess_conv3d_input(x, data_format):
    """Transpose and cast the input before the conv3d.
    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    """
    # tensorflow doesn't support float64 for conv layer before 1.8.0
    if (dtype(x) == 'float64' and
            StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.8.0')):
        x = tf.cast(x, 'float32')
    tf_data_format = 'NDHWC'
    return x, tf_data_format


def depthwise_conv3d_args_preprocessor(args, kwargs):
    converted = []

    if 'init' in kwargs:
        init = kwargs.pop('init')
        kwargs['depthwise_initializer'] = init
        converted.append(('init', 'depthwise_initializer'))

    args, kwargs, _converted = conv3d_args_preprocessor(args, kwargs)
    return args, kwargs, converted + _converted

    legacy_depthwise_conv3d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=depthwise_conv3d_args_preprocessor)

# Implementation: https://github.com/alexandrosstergiou/keras-DepthwiseConv3D

class DepthwiseConv3D(Conv3D):
    """Depthwise 3D convolution.
    Depth-wise part of separable convolutions consist in performing
    just the first step/operation
    (which acts on each input channel separately).
    It does not perform the pointwise convolution (second step).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.
    # Arguments
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, width and height of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along the depth, width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filterss_in * depth_multiplier`.
        groups: The depth size of the convolution (as a variant of the original Depthwise conv)
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        dialation_rate: List of ints.
                        Defines the dilation factor for each dimension in the
                        input. Defaults to (1,1,1)
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        5D tensor with shape:
        `(batch, depth, channels, rows, cols)` if data_format='channels_first'
        or 5D tensor with shape:
        `(batch, depth, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        5D tensor with shape:
        `(batch, filters * depth, new_depth, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_depth, new_rows, new_cols, filters * depth)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 groups=None,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 dilation_rate = (1, 1, 1),
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv3D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            dilation_rate=dilation_rate,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.groups = groups
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.dilation_rate = dilation_rate
        self._padding = _preprocess_padding(self.padding)
        self._strides = (1,) + self.strides + (1,)
        self._data_format = "NDHWC"
        self.input_dim = None

    def build(self, input_shape):
        if len(input_shape) < 5:
            raise ValueError('Inputs to `DepthwiseConv3D` should have rank 5. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv3D` '
                             'should be defined. Found `None`.')
        self.input_dim = int(input_shape[channel_axis])

        if self.groups is None:
            self.groups = self.input_dim

        if self.groups > self.input_dim:
            raise ValueError('The number of groups cannot exceed the number of channels')

        if self.input_dim % self.groups != 0:
            raise ValueError('Warning! The channels dimension is not divisible by the group size chosen')

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  self.kernel_size[2],
                                  self.input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.groups * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs, training=None):
        inputs = _preprocess_conv3d_input(inputs, self.data_format)

        if self.data_format == 'channels_last':
            dilation = (1,) + self.dilation_rate + (1,)
        else:
            dilation = self.dilation_rate + (1,) + (1,)

        if self._data_format == 'NCDHW':
            outputs = tf.concat(
                [tf.nn.conv3d(inputs[0][:, i:i+self.input_dim//self.groups, :, :, :], self.depthwise_kernel[:, :, :, i:i+self.input_dim//self.groups, :],
                    strides=self._strides,
                    padding=self._padding,
                    dilations=dilation,
                    data_format=self._data_format) for i in range(0, self.input_dim, self.input_dim//self.groups)], axis=1)

        else:
            outputs = tf.concat(
                [tf.nn.conv3d(inputs[0][:, :, :, :, i:i+self.input_dim//self.groups], self.depthwise_kernel[:, :, :, i:i+self.input_dim//self.groups, :],
                    strides=self._strides,
                    padding=self._padding,
                    dilations=dilation,
                    data_format=self._data_format) for i in range(0, self.input_dim, self.input_dim//self.groups)], axis=-1)

        if self.bias is not None:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            depth = input_shape[2]
            rows = input_shape[3]
            cols = input_shape[4]
            out_filters = self.groups * self.depth_multiplier
        elif self.data_format == 'channels_last':
            depth = input_shape[1]
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = self.groups * self.depth_multiplier

        depth = conv_utils.conv_output_length(depth, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])

        rows = conv_utils.conv_output_length(rows, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        cols = conv_utils.conv_output_length(cols, self.kernel_size[2],
                                             self.padding,
                                             self.strides[2])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, depth, rows, cols)

        elif self.data_format == 'channels_last':
            return (input_shape[0], depth, rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv3D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config

DepthwiseConvolution3D = DepthwiseConv3D

backend = None
layers = None
models = None
keras_utils = None

### Model Base https://github.com/qubvel/efficientnet


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1,1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2,2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2,2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2,2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1,1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2,2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1,1], se_ratio=0.25)
]


## Our MB_CONV_Block
def mb_conv_block(inputs,block_args,drop_rate):
  ##Mobile Inverted Residual block along with Squeeze  and Excitation block.
    kernel_size = block_args.kernel_size
    num_repeat= block_args.num_repeat
    input_filters= block_args.input_filters
    output_filters=block_args. output_filters
    expand_ratio= block_args.expand_ratio
    id_skip= block_args.id_skip
    strides= block_args.strides
    se_ratio= block_args.se_ratio
  # expansion phase

    expanded_filters = input_filters * expand_ratio
    x=tf.keras.layers.Conv3D(filters=expanded_filters, kernel_size=(1,1, 1), padding="same",use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,)(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.activations.swish(x)

    # Depthwise convolution phase

    x_depth=DepthwiseConv3D(kernel_size=kernel_size, padding="same",strides=strides, use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,)(x) 
    x=tf.keras.layers.BatchNormalization()(x_depth)
    x=tf.keras.activations.swish(x)

    #SE Block
    x =tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Reshape((1,1, 1, expanded_filters ))(x)
    squeezed_filters = max (1, int(input_filters * se_ratio))
    x=tf.keras.layers.Conv3D(filters=squeezed_filters, kernel_size=(1,1, 1),padding="same",kernel_initializer=CONV_KERNEL_INITIALIZER,)(x)
    x=tf.keras.activations.swish(x)
    x=tf.keras.layers.Conv3D(filters=expanded_filters, kernel_size=(1,1, 1),padding="same",kernel_initializer=CONV_KERNEL_INITIALIZER,)(x)
    x=tf.keras.activations.sigmoid(x)
    x=tf.keras.layers.Multiply()([x_depth,x])
    #SE Block
    x=tf.keras.layers.Conv3D(filters=output_filters, kernel_size=(1,1, 1),padding="same",use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,)(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Dropout(drop_rate)(x)

    if id_skip and all( s == 1 for s in strides) and input_filters == output_filters:

      x=tf.keras.layers.Add()([inputs,x])
  
    return x


def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))



def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 model_name='efficientnet',
                 weights='imagenet',
                 input_shape=None,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 **kwargs):
  

  #### Stem
  inputs = tf.keras.layers.Input(shape=(input_shape))
  x = tf.keras.layers.Conv3D(round_filters(32, width_coefficient, depth_divisor), 3,
                      strides=(2, 2,2),
                      padding='same',
                      use_bias=False,
                      name='stem_conv',kernel_initializer=CONV_KERNEL_INITIALIZER,)(inputs)
  x = tf.keras.layers.BatchNormalization( name='stem_bn')(x)
  x=tf.keras.activations.swish(x)
  num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
  block_num = 0
  for idx, block_args in enumerate(blocks_args):
      assert block_args.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      block_args = block_args._replace(
          input_filters=round_filters(block_args.input_filters,
                                      width_coefficient, depth_divisor),
          output_filters=round_filters(block_args.output_filters,
                                        width_coefficient, depth_divisor),
          num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

      # The first block needs to take care of stride and filter size increase.
      drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
      x = mb_conv_block(x, block_args,
                        drop_rate=drop_rate)
      block_num += 1
      if block_args.num_repeat > 1:
          # pylint: disable=protected-access
          block_args = block_args._replace(
              input_filters=block_args.output_filters, strides=(1, 1,1))
          # pylint: enable=protected-access
          for bidx in xrange(block_args.num_repeat - 1):
              drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
              block_prefix = 'block{}{}_'.format(
                  idx + 1,
                  string.ascii_lowercase[bidx + 1]
              )
              x = mb_conv_block(x, block_args,
                                drop_rate=drop_rate)
              block_num += 1
  x = tf.keras.layers.Conv3D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',
                      use_bias=False,
                      name='top_conv',kernel_initializer=CONV_KERNEL_INITIALIZER,)(x)
  x = tf.keras.layers.BatchNormalization(name='top_bn')(x)
  x=tf.keras.activations.swish(x)
  return tf.keras.Model(inputs, x, name=model_name)




def EfficientNet3DB0(
        input_tensor=None,
        input_shape=None,
        **kwargs
):
    return EfficientNet(
        1.0, 1.0, 224, 0.2,
        model_name='efficientnet3d-b0',
        input_tensor=input_tensor, input_shape=input_shape,
        **kwargs
    )


def EfficientNet3DB1(
        input_tensor=None,
        input_shape=None,
        **kwargs
        ):
    return EfficientNet(
        1.0, 1.1, 240, 0.2,
        model_name='efficientnet3d-b1',
        input_tensor=input_tensor, input_shape=input_shape,
        **kwargs
    )


def EfficientNet3DB2(
                   input_tensor=None,
                   input_shape=None,
                   **kwargs
                   ):
    return EfficientNet(
        1.1, 1.2, 260, 0.3,
        model_name='efficientnet3d-b2',
        input_tensor=input_tensor, input_shape=input_shape,
        **kwargs
    )


def EfficientNet3DB3(
                   input_tensor=None,
                   input_shape=None,**kwargs
                   ):
    return EfficientNet(
        1.2, 1.4, 300, 0.3,
        model_name='efficientnet3d-b3',
        input_tensor=input_tensor, input_shape=input_shape,
        **kwargs
    )


def EfficientNet3DB4(
        input_tensor=None,
        input_shape=None,
        **kwargs
        ):
    return EfficientNet(
        1.4, 1.8, 380, 0.4,
        model_name='efficientnet-b4',
        input_tensor=input_tensor, input_shape=input_shape,
        **kwargs
    )


def EfficientNet3DB5(
        input_tensor=None,
        input_shape=None,
        **kwargs
        ):
    return EfficientNet(
        1.6, 2.2, 456, 0.4,
        model_name='efficient3dnet-b5',
        input_tensor=input_tensor, input_shape=input_shape,
        **kwargs
    )


def EfficientNet3DB6(
        input_tensor=None,
        input_shape=None,
        **kwargs
        ):
    return EfficientNet(
        1.8, 2.6, 528, 0.5,
        model_name='efficientnet3d-b6',
        input_tensor=input_tensor, input_shape=input_shape,

        **kwargs
    )


def EfficientNet3DB7(
        input_tensor=None,
        input_shape=None,
        **kwargs
        ):
    return EfficientNet(
        2.0, 3.1, 600, 0.5,
        model_name='efficientnet3d-b7',
        input_tensor=input_tensor, input_shape=input_shape,
        **kwargs
    )


def EfficientNet3DL2(
        input_tensor=None,
        input_shape=None,
        **kwargs

        ):
    return EfficientNet(
        4.3, 5.3, 800, 0.5,
        model_name='efficientnet3d-l2',
        input_tensor=input_tensor, input_shape=input_shape, 
        **kwargs
    )
