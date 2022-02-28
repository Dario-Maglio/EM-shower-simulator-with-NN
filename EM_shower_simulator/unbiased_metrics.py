""" Definition of unbiased metrics based on physical properties"""

import numpy as np
import tensorflow as tf

#-------------------------------------------------------------------------------
ENERGY_NORM = 6.7404
ENERGY_SCALE = 1000000
#-------------------------------------------------------------------------------
    # shower_depth_mean, shower_depth_std = shower_depth(train_images)
    # print(f"Mean shower depth = {shower_depth_mean}, std = {shower_depth_std}")

def shower_depth_lateral_width(showers_vector):
    """Compute shower mean depth and mean lateral width among layers and std."""
    shape = showers_vector.shape

    layer_num= tf.constant([[x for x in range(shape[1])]])
    layer_num= tf.cast(tf.tile(layer_num, [shape[0],1] ), tf.float32)
    pixel_num= tf.constant([[[[x for x in range(-shape[2]//2+1, shape[2]//2+1)]
                            for y in range(-shape[2]//2+1, shape[2]//2+1)]
                            for l in range(shape[1]) ]])
    pixel_num= tf.cast(tf.tile(
                        pixel_num, [shape[0],1,1,1] ), tf.float32)
    pixel_num= tf.reshape(pixel_num, shape)

    pixel_en = tf.math.multiply(showers_vector, ENERGY_NORM)
    pixel_en = tf.math.pow(10., pixel_en)
    pixel_en = tf.math.divide(pixel_en, ENERGY_SCALE)

    layers_en = tf.math.reduce_sum(pixel_en, axis=[2,3,4])
    total_en  = tf.math.reduce_sum(layers_en, axis=1)

    layers_scalar_prod_en   = tf.math.multiply(layers_en, layer_num)
    depth_weighted_total_en = tf.math.reduce_sum(layers_scalar_prod_en, axis=1)

    # shower depth
    shower_depth      = tf.math.divide(depth_weighted_total_en,total_en)
    shower_depth_mean = tf.math.reduce_mean(shower_depth, axis = 0)
    shower_depth_std  = tf.math.reduce_std(shower_depth, axis=0)

    x = tf.math.multiply(pixel_en,pixel_num)
    x = tf.math.reduce_sum(x, axis=[2,3,4])

    x2 = tf.math.multiply(pixel_en, pixel_num**2)
    x2 = tf.math.reduce_sum(x2, axis=[2,3,4])

    # shower lateral width
    lateral_width      = tf.math.sqrt(tf.math.abs(x2/layers_en - (x/layers_en)**2))
    lateral_width_mean = tf.math.reduce_mean(lateral_width, axis=[0,1])
    lateral_width_std  = tf.math.reduce_std(lateral_width, axis=[0,1])

    return {"shower mean depth":shower_depth_mean,
            "shower std depth":shower_depth_std,
            "mean lateral width":lateral_width_mean,
            "std lateral width":lateral_width_std
            }
