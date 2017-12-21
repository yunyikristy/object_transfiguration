
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

        with tf.variable_scope("d_net"):
            self.d_encode_img_template = self.d_encode_image()
            self.d_context_template = self.context_embedding()
            self.discriminator_template = self.discriminator()

    def build(self, c_var, z, c_mask):

        #with tf.variable_scope("g_net"):
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

    # d-net
    def context_embedding(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template

    def d_encode_image(self):
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1

    def discriminator(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))

        return template

    def get_discriminator(self, x_var, c_text, c_var):
        x_code = self.d_encode_img_template.construct(input=x_var)
        c_code = self.d_encode_img_template.construct(input=c_var)
        c_text_code = self.d_context_template.construct(input=c_text)
        c_text_code = tf.expand_dims(tf.expand_dims(c_text_code, 1), 1)
        c_text_code = tf.tile(c_text_code, [1, self.s16, self.s16, 1])
        x_c_code = tf.concat(3, [x_code, c_code])
        x_c_code = tf.concat(3, [x_c_code, c_text_code])
        return self.discriminator_template.construct(input=x_c_code)

    
    def compute_losses(self, images, wrong_images, fake_images, embeddings, conditions):


        real_logit = self.get_discriminator(images, embeddings, conditions)
        wrong_logit = self.get_discriminator(wrong_images, embeddings, conditions)
        fake_logit = self.get_discriminator(fake_images, embeddings, conditions)

        real_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(real_logit,
                                                    tf.ones_like(real_logit))
        real_d_loss = tf.reduce_mean(real_d_loss)
        wrong_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(wrong_logit,
                                                    tf.zeros_like(wrong_logit))
        wrong_d_loss = tf.reduce_mean(wrong_d_loss)
        fake_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(fake_logit,
                                                    tf.zeros_like(fake_logit))
        fake_d_loss = tf.reduce_mean(fake_d_loss)
        discriminator_loss =\
                real_d_loss + (wrong_d_loss + fake_d_loss) / 2.

        generator_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(fake_logit,
                                                    tf.ones_like(fake_logit))
        generator_loss = tf.reduce_mean(generator_loss)

        return discriminator_loss, generator_loss


