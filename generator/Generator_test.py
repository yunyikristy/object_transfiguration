
import prettytensor as pt
import tensorflow as tf
import misc.custom_ops
from misc.custom_ops import leaky_rectify


class Generator:
    def __init__(self):
        self.gf_dim = 128
        self.df_dim = 64
        self.ef_dim = 128

        self.s = 64
        self.s2, self.s4, self.s8, self.s16 =\
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

    def build(self, images, c_var, z, c_mask):

        #images = images * 2. / 255. - 1
        #images = tf.image.resize_images(images, [64,64])
        #c_mask = c_mask * 2. / 255. - 1
        #c_mask = tf.concat(3, [c_mask, c_mask, c_mask])
        #c_mask = tf.image.resize_images(c_mask, [64,64])
        g_input = self.generate_condition(tf.concat(1, [c_var, z]), c_mask)
        fake_images = self.generator(g_input)
        return fake_images

    def generate_condition(self, c_var, c_mask):
        condition_vec =\
                (pt.wrap(c_var).
                flatten().
                custom_fully_connected(4096).
                reshape([-1, 8, 8, 64]).
                custom_conv2d(64, 5, 5, 1, 1).
                conv_batch_norm().
                apply(tf.nn.relu).
                custom_conv2d(64, 5, 5, 1, 1).
                conv_batch_norm().
                apply(tf.nn.relu))

        condition_mask =\
                (pt.wrap(c_mask).
                custom_conv2d(64, 5, 5, 2, 2).
                conv_batch_norm().
                apply(tf.nn.relu).
                custom_conv2d(64, 5, 5, 2, 2).
                conv_batch_norm().
                apply(tf.nn.relu).
                custom_conv2d(128, 5, 5, 2, 2).
                conv_batch_norm().
                apply(tf.nn.relu))

        condition = tf.concat(3, [condition_vec, condition_mask])
        return condition
        
    def generator(self, z_var):
        node1_0 =\
            (pt.wrap(z_var).
            custom_conv2d(self.gf_dim * 8, 5, 5, 2, 2).
            conv_batch_norm().
            apply(tf.nn.relu))
        node1_1 = \
            (node1_0.
             custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(tf.nn.relu))

        node2_0 = \
            (node1.
             apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2_1 = \
            (node2_0.
             custom_conv2d(self.gf_dim * 1, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 1, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2 = \
            (node2_0.
             apply(tf.add, node2_1).
             apply(tf.nn.relu))

        output_tensor = \
            (node2.
             apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor


