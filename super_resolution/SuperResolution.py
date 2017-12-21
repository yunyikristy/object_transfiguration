import prettytensor as pt
import tensorflow as tf
import misc.custom_ops
from misc.custom_ops import leaky_rectify

def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss

class SuperResolution(object):
    def __init__(self):
        self.gf_dim = 128
        self.df_dim = 64
        self.ef_dim = 128

        self.s = 64
        self.s2, self.s4, self.s8, self.s16 = \
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

        with tf.variable_scope("hr_d_net"):
            self.hr_d_context_template = self.context_embedding()
            self.hr_d_image_template = self.hr_d_encode_image()
            self.hr_discriminator_template = self.hr_discriminator()

    def build(self, input_, c_var):
        fake_images = self.hr_get_generator(input_, c_var)
        return fake_images

    def hr_generate_condition(self, c_var):
        conditions =\
            (pt.wrap(c_var).
             flatten().
             custom_fully_connected(self.ef_dim * 2).
             apply(leaky_rectify, leakiness=0.2))
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        return [mean, log_sigma]

    # stage II generator (hr_g)
    def residual_block(self, x_c_code):
        node0_0 = pt.wrap(x_c_code)  # -->s4 * s4 * gf_dim * 4
        node0_1 = \
            (pt.wrap(x_c_code).  # -->s4 * s4 * gf_dim * 4
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        output_tensor = \
            (node0_0.
             apply(tf.add, node0_1).
             apply(tf.nn.relu))
        return output_tensor

    def hr_g_encode_image(self, x_var):
        output_tensor = \
            (pt.wrap(x_var).  # -->s * s * 3
             custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).  # s * s * gf_dim
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=4, k_w=4).  # s2 * s2 * gf_dim * 2
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=4, k_w=4).  # s4 * s4 * gf_dim * 4
             conv_batch_norm().
             apply(tf.nn.relu))
        return output_tensor

    def hr_g_joint_img_text(self, x_c_code):
        output_tensor = \
            (pt.wrap(x_c_code).  # -->s4 * s4 * (ef_dim+gf_dim*4)
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).  # s4 * s4 * gf_dim * 4
             conv_batch_norm().
             apply(tf.nn.relu))
        return output_tensor

    def hr_generator(self, x_c_code):
        output_tensor = \
            (pt.wrap(x_c_code).  # -->s4 * s4 * gf_dim*4
             # custom_deconv2d([0, self.s2, self.s2, self.gf_dim * 2], k_h=4, k_w=4).  # -->s2 * s2 * gf_dim*2
             apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s, self.s, self.gf_dim], k_h=4, k_w=4).  # -->s * s * gf_dim
             apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s * 2, self.s * 2, self.gf_dim // 2], k_h=4, k_w=4).  # -->2s * 2s * gf_dim/2
             apply(tf.image.resize_nearest_neighbor, [self.s * 2, self.s * 2]).
             custom_conv2d(self.gf_dim // 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s * 4, self.s * 4, self.gf_dim // 4], k_h=4, k_w=4).  # -->4s * 4s * gf_dim//4
             apply(tf.image.resize_nearest_neighbor, [self.s * 4, self.s * 4]).
             custom_conv2d(self.gf_dim // 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).  # -->4s * 4s * 3
             apply(tf.nn.tanh))
        return output_tensor

    def hr_get_generator(self, x_var, c_code):
        # image x_var: self.s * self.s *3
        x_code = self.hr_g_encode_image(x_var)  # -->s4 * s4 * gf_dim * 4

        # text c_code: ef_dim
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = \
            (pt.wrap(c_code).
            flatten().
            custom_fully_connected(4096).
            reshape([-1, 16, 16, 16]).
            custom_conv2d(16, 5, 5, 1, 1).
            conv_batch_norm().
            apply(tf.nn.relu).
            custom_conv2d(16, 5, 5, 1, 1).
            conv_batch_norm().
            apply(tf.nn.relu))

        # combine both --> s4 * s4 * (ef_dim+gf_dim*4)
        x_c_code = tf.concat(3, [x_code, c_code])
        # Joint learning from text and image -->s4 * s4 * gf_dim * 4
        node0 = self.hr_g_joint_img_text(x_c_code)
        node1 = self.residual_block(node0)
        node2 = self.residual_block(node1)
        node3 = self.residual_block(node2)
        node4 = self.residual_block(node3)

        # Up-sampling
        return self.hr_generator(node4)  # -->4s * 4s * 3

    # d and hr_d build this structure separately and do not share parameters
    def context_embedding(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template


    # hr_d_net
    def hr_d_encode_image(self):
        node1_0 = \
            (pt.template("input").  # 4s * 4s * 3
             custom_conv2d(self.df_dim, k_h=4, k_w=4).  # 2s * 2s * df_dim
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).  # s * s * df_dim*2
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).  # s2 * s2 * df_dim*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).  # s4 * s4 * df_dim*8
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 16, k_h=4, k_w=4).  # s8 * s8 * df_dim*16
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 32, k_h=4, k_w=4).  # s16 * s16 * df_dim*32
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 16, k_h=1, k_w=1, d_h=1, d_w=1).  # s16 * s16 * df_dim*16
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # s16 * s16 * df_dim*8
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

    def hr_get_discriminator(self, x_var, c_var):
        x_code = self.hr_d_image_template.construct(input=x_var)
        c_code = self.hr_d_context_template.construct(input=c_var)
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, self.s16, self.s16, 1])
        x_c_code = tf.concat(3, [x_code, c_code])
        return self.hr_discriminator_template.construct(input=x_c_code)

    def hr_discriminator(self):
        template = \
            (pt.template("input").
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(3, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))
        return template

    def compute_losses(self, images, wrong_images, fake_images, embeddings):
        real_logit =\
            self.hr_get_discriminator(images, embeddings)
        wrong_logit =\
            self.hr_get_discriminator(wrong_images, embeddings)
        fake_logit =\
            self.hr_get_discriminator(fake_images, embeddings)

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

    def sample_encoded_context(self, embeddings):
        c_mean_logsigma = self.hr_generate_condition(embeddings)
        mean = c_mean_logsigma[0]

        epsilon = tf.truncated_normal(tf.shape(mean))
        stddev = tf.exp(c_mean_logsigma[1])
        c = mean + stddev * epsilon
        kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        return c, 2 * kl_loss
