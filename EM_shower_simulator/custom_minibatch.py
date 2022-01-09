""" Taken from
https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py
"""
import tensorflow as tf
import numpy as np

MBSTD_GROUP_SIZE = 6
test_noise = tf.random.normal([3, 1, 10, 10, 3], stddev=1.0)
test_noise = tf.concat([test_noise, tf.random.normal([3, 1, 10, 10, 3], stddev=2.0)], axis=1)
test_noise = tf.concat([test_noise, tf.random.normal([3, 1, 10, 10, 3], stddev=3.0)], axis=1)

# Minibatch standard deviation.

def minibatch_stddev_layer(discr, group_size=MBSTD_GROUP_SIZE):
    """Minibatch discrimination layer is important to avoid mode collapse.
    Once it is wrapped with a Lambda Keras layer it returns an additional filter
    node with information about the statistical distribution of the group_size,
    allowing the discriminator to recognize when the generator strarts to
    replicate the same kind of event multiple times.

    Inspired by
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py
    """
    with tf.compat.v1.variable_scope('MinibatchStddev'):
        # Input 0 dimension must be divisible by (or smaller than) group_size.
        group_size = tf.minimum(group_size, tf.shape(discr)[0])
        # Input shape.
        shape = discr.shape
        # Split minibatch into M groups of size G.
        minib = tf.reshape(discr, [group_size, -1, shape[1], shape[2], shape[3], shape[4]])
        # Cast to FP32.
        minib = tf.cast(minib, tf.float32)
        # Subtract mean over group.
        minib = minib - tf.reduce_mean(minib, axis=0, keepdims=True)
        print(minib.shape)
        # Calculate variance over group.
        minib = tf.reduce_mean(tf.square(minib), axis=0)
        # Calculate std dev over group.
        minib = tf.sqrt(minib + 1e-8)
        print(minib.shape)
        # Take average over fmaps and pixels.
        minib = tf.reduce_mean(minib, axis=[1,2,3,4], keepdims=True)
        # Cast back to original data type.
        minib = tf.cast(minib, discr.dtype)
        # New tensor by replicating input multiples times.
        minib = tf.tile(minib, [group_size, shape[1], shape[2], shape[3], 1])
        # Append as new fmap.
        return tf.concat([discr, minib], axis=-1)


def minibatch_stddev_layer_2(discr, group_size=MBSTD_GROUP_SIZE):
    """Minibatch discrimination layer is important to avoid mode collapse.
    Once it is wrapped with a Lambda Keras layer it returns an additional filter
    node with information about the statistical distribution of the group_size,
    allowing the discriminator to recognize when the generator strarts to
    replicate the same kind of event multiple times.
    Inspired by
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py
    """
    with tf.compat.v1.variable_scope('MinibatchStddev'):
      # Minibatch/batch must be divisible by (or smaller than) group_size.
      group_size = tf.minimum(group_size, tf.shape(discr)[0])
      # Input shape.
      shape = discr.shape
      # Split minibatch into N categories, one for each layer
      minib = tf.reshape(discr, [group_size, -1, shape[1], shape[2], shape[3], shape[4]])
      for layer in range(shape[1]):
        minib_layer = minib[:, :, layer, :, :, :]
        # Cast to FP32.
        minib_layer = tf.cast(minib_layer, tf.float32)
        # Subtract mean over group.
        minib_layer = minib_layer - tf.reduce_mean(minib_layer, axis=0, keepdims=True)
        # Calculate variance over group.
        minib_layer = tf.reduce_mean(tf.square(minib_layer), axis=0)
        # Calculate std dev over group.
        minib_layer = tf.sqrt(minib_layer + 1e-8)
        # Take average over fmaps and pixels.
        minib_layer = tf.reduce_mean(minib_layer, axis=[1,2,3], keepdims=True)
        # Cast back to original data type.
        minib_layer = tf.cast(minib_layer, discr.dtype)
        # New tensor by replicating input multiples times.
        minib_layer = tf.tile(minib_layer, [group_size, 1, shape[2], shape[3]])
        # Concatenate minib layer to minib
        if(layer==0):
          layers_minib = minib_layer
        else:
          layers_minib = tf.concat([layers_minib, minib_layer], axis = 1)
    layers_minib = tf.reshape(layers_minib, [-1, shape[1] , shape[2], shape[3], 1])
    return tf.concat([discr, layers_minib], axis=-1)

def minibatch_stddev_layer_3(discr, group_size=MBSTD_GROUP_SIZE):
    """Minibatch discrimination layer is important to avoid mode collapse.
    Once it is wrapped with a Lambda Keras layer it returns an additional filter
    node with information about the statistical distribution of the group_size,
    allowing the discriminator to recognize when the generator strarts to
    replicate the same kind of event multiple times.

    Inspired by
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py
    """
    with tf.compat.v1.variable_scope('MinibatchStddev'):
        # Input 0 dimension must be divisible by (or smaller than) group_size.
        group_size = tf.minimum(group_size, tf.shape(discr)[0])
        # Input shape.
        shape = discr.shape
        print(shape)
        # Split minibatch into M groups of size G.
        minib = tf.reshape(discr, [group_size, -1, shape[1], shape[2], shape[3], shape[4]])
        print(minib.shape)
        # Cast to FP32.
        minib = tf.cast(minib, tf.float32)
        # Calculate the std deviation for each pixel over minibatch.
        minib = tf.math.reduce_std(minib, axis=0)
        print(minib.shape)
        # Take average over pixels to get a kind of fmap std deviation.
        minib = tf.reduce_mean(minib, axis=[1,2,3], keepdims=True)
        print(minib.shape)
        # Cast back to original data type.
        minib = tf.cast(minib, discr.dtype)
        # New tensor by replicating input multiples times.
        minib = tf.tile(minib, [group_size, shape[1], shape[2], shape[3], 1])
        print(minib.shape)
        # Append as new fmap.
        return tf.concat([discr, minib], axis=-1)

if __name__ == "__main__":
   minibatch_stddev_layer(test_noise)[:, :, :, :, -1]
   minibatch_stddev_layer_3(test_noise)[:, :, :, :, -1]
   #print(tf.math.reduce_std(test_noise, axis=1))



"""
#----------------------------------------------------------------------------
# Discriminator network used in the paper.

def D_paper(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    x = downscale2d(x)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            return x

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out
"""
#----------------------------------------------------------------------------
