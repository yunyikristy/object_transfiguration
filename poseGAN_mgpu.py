#-*-coding:utf-8-*-
import tensorflow as tf
import prettytensor as pt
import numpy as np
import cv2
import pickle
import scipy.misc
from app_encoder.AppEncoder import AppEncoder
from generator.Generator import Generator
from super_resolution.SuperResolution import SuperResolution
import time

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars if g != None]
        #print grads
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def average_losses(loss):

    tf.add_to_collection('losses', loss)
    losses = tf.get_collection('losses')
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name="total_loss")
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


class PoseGAN:

    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args["batch_size"]
        self.image_size_hr = args["image_size_hr"]
        self.image_size_lr = args["image_size_lr"]
        self.lr = args["lr"]
        self.epoch_num = args["epoch_num"]

        with tf.variable_scope("app_encoder"):
            self.app_encoder = AppEncoder()

        with tf.variable_scope("generator"):
            self.generator = Generator()

        with tf.variable_scope("super_resolution"):
            self.super_resolution = SuperResolution()

        print ('init success')

    def load_weights(self):
        tvars = tf.all_variables()
        app_encoder_weights = np.load("../final/pretrained/app_encoder.npy").item()
        generator_weights = np.load("../final/pretrained/generator.npy").item()
        ops = []
        for var in tvars:
            if var.name.startswith("app_encoder/"):
                try:
                    ops.append(var.assign(app_encoder_weights[var.name]))
                except:
                    print var.name
                    pass
            elif var.name.startswith("generator/"):
                try:
                    ops.append(var.assign(generator_weights[var.name]))
                except:
                    print var.name
                    pass

        self.sess.run(ops)

    def load_data(self):
        f = open('../final/Data/train/304images.pickle', 'rb')
        self.images = pickle.load(f)
        f.close()
        self.images = np.array(self.images)
        self.num_examples = self.images.shape[0]

        self.start_idx = 0
        self.end_idx = self.batch_size

        print('load image success')

        f = open('../final/Data/train/class_info.pickle', 'rb')
        tmp_class_id = pickle.load(f)
        tmp_class_id = np.array(tmp_class_id)
        self.class_id = np.zeros((tmp_class_id.shape[0], 200), dtype=np.float32)
        for i in range(tmp_class_id.shape[0]):
            self.class_id[i, tmp_class_id[i]-1] = 1
        print self.class_id.shape
        f.close()

        print ('load class info success')

        f = open('../final/Data/train/76segs.pickle', 'rb')
        self.segs = pickle.load(f)
        f.close()
        self.segs = np.array(self.segs)

        f = open('../final/Data/train/googlenet.pickle', 'rb')
        self.googlenet = pickle.load(f)
        f.close()

        self.shuffle_ids = np.arange(self.num_examples)
        #np.random.shuffle(self.shuffle_ids)

    def train(self):

        with tf.device('/cpu:0'):
            g_opt = tf.train.AdamOptimizer(self.lr, beta1=0.5)
            d_opt = tf.train.AdamOptimizer(self.lr, beta1=0.5)
            f_opt = tf.train.AdamOptimizer(self.lr, beta1=0.5)

            models = []
            num_gpu = 2
            for gpu_id in range(num_gpu):
                with tf.device('/gpu:%d'%gpu_id):
                    with tf.name_scope('tower_%d'%gpu_id):
                        with tf.variable_scope('cpu_variables', reuse=gpu_id>0):
                            src_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size_hr, self.image_size_hr, 3],
                                            name = "source_images")

                            src_pose = tf.placeholder(tf.float32, [self.batch_size, self.image_size_lr, self.image_size_lr, 3],
                                            name = "source_pose")
                            trg_pose = tf.placeholder(tf.float32, [self.batch_size, self.image_size_lr, self.image_size_lr, 3],
                                            name = "target_pose")

                            src_app = tf.placeholder(tf.float32, [self.batch_size, 1024], name = "source_appearance")

                            pix_lambda = tf.placeholder(tf.float32, [])

                        with tf.variable_scope('generator', reuse=gpu_id>0):
                            z = tf.random_normal(shape=(self.batch_size, 100))
                            with tf.variable_scope("g_net", reuse=gpu_id>0):
                                fake_s2s_stageI = self.generator.build(src_app, z, src_pose)
                            with tf.variable_scope("g_net", reuse=True):
                                fake_s2t_stageI = self.generator.build(src_app, z, trg_pose)

                        with tf.variable_scope("super_resolution"):
                            with tf.variable_scope("hr_g_net", reuse=gpu_id>0):
                                fake_s2s_stageII = self.super_resolution.build(fake_s2s_stageI, src_app)
                            with tf.variable_scope("hr_g_net", reuse=True):
                                fake_s2t_stageII = self.super_resolution.build(fake_s2t_stageI, src_app)

                        with tf.variable_scope("app_encoder", reuse=gpu_id>0):
                            #tmp_fake_s2s = (fake_s2s_stageII + 1) * (255. / 2)
                            tmp_fake_s2t = (fake_s2t_stageII + 1) * (255. / 2)
                            #fake_s2s_app = self.app_encoder.build(tmp_fake_s2s)
                            fake_s2t_app = self.app_encoder.build(tmp_fake_s2t)

                        with tf.variable_scope("loss"):
                            const_loss = tf.reduce_mean(tf.square(src_app - fake_s2t_app))
                            pix_loss = tf.reduce_mean(tf.abs(src_images - fake_s2s_stageII))
    
                            # regular d_loss and gan_loss
                            real_logit_s = self.super_resolution.hr_get_discriminator(src_images, src_app)
    
                            fake_logit_s2t = self.super_resolution.hr_get_discriminator(fake_s2t_stageII, src_app)
                            fake_logit_s2s = self.super_resolution.hr_get_discriminator(fake_s2s_stageII, src_app)
    
                            #s2tloss
                            fake_d_loss_s2s =\
                                    tf.nn.softmax_cross_entropy_with_logits(fake_logit_s2s,
                                                                            tf.constant([[0.0, 0.0, 1.0]] * self.batch_size))
                            fake_d_loss_s2s = tf.reduce_mean(fake_d_loss_s2s)
                            fake_d_loss_s2t =\
                                    tf.nn.softmax_cross_entropy_with_logits(fake_logit_s2t,
                                                                            tf.constant([[0.0, 1.0, 0.0]] * self.batch_size))
                            fake_d_loss_s2t = tf.reduce_mean(fake_d_loss_s2t)
                            real_d_loss =\
                                    tf.nn.softmax_cross_entropy_with_logits(real_logit_s,
                                                                            tf.constant([[1.0, 0.0, 0.0]] * self.batch_size))
                            real_d_loss = tf.reduce_mean(real_d_loss)
    
                            gan_loss_s2s =\
                                    tf.nn.softmax_cross_entropy_with_logits(fake_logit_s2s,
                                                                            tf.constant([[1.0, 0.0, 0.0]] * self.batch_size))
                            gan_loss_s2s = tf.reduce_mean(gan_loss_s2s)
                            gan_loss_s2t =\
                                    tf.nn.softmax_cross_entropy_with_logits(fake_logit_s2t,
                                                                            tf.constant([[1.0, 0.0, 0.0]] * self.batch_size))
                            gan_loss_s2t = tf.reduce_mean(gan_loss_s2t)
    
                            d_loss = fake_d_loss_s2s / 2. + real_d_loss + fake_d_loss_s2t / 2.
                            g_loss = (gan_loss_s2s + gan_loss_s2t) / 2. + 10. * pix_loss + const_loss / 2.

                        t_vars = tf.trainable_variables()
                        g_train_vars = []
                        d_train_vars = []

                        for var in t_vars:
                            if var.name.startswith('super_resolution'):
                                if 'g_net' in var.name:
                                    g_train_vars.append(var)
                                elif 'd_net' in var.name:
                                    d_train_vars.append(var)
                            #elif var.name.startswith('app_encoder/'):
                            #    f_train_vars.append(var)
                            #    if not 'loss' in var.name:
                            #        g_train_vars.append(var)

                        d_grad = d_opt.compute_gradients(d_loss, var_list=d_train_vars)
                        g_grad = g_opt.compute_gradients(g_loss, var_list=g_train_vars)
                        #f_grad = f_opt.compute_gradients(f_loss, var_list=f_train_vars)

                        models.append((src_images, src_pose, trg_pose, src_app, pix_lambda, \
                            fake_s2s_stageI, fake_s2t_stageI, fake_s2s_stageII, fake_s2t_stageII, \
                            d_loss, d_grad, g_loss, g_grad))
                            #f_loss, f_grad))


            print('build model on gpu tower done.')

            _, _, _, _, _, \
            _, _, _, _,\
            tower_d_loss, tower_d_grad, tower_g_loss, tower_g_grad \
            = zip(*models)

            #tower_f_loss, tower_f_grad \

            aver_d_loss = tf.reduce_mean(tower_d_loss)
            aver_g_loss = tf.reduce_mean(tower_g_loss)
            #aver_f_loss = tf.reduce_mean(tower_f_loss)

            d_op = d_opt.apply_gradients(average_gradients(tower_d_grad))
            g_op = g_opt.apply_gradients(average_gradients(tower_g_grad))
            #f_op = f_opt.apply_gradients(average_gradients(tower_f_grad))

            with tf.control_dependencies([d_op, g_op]):
                all_op = tf.no_op()

            print('done')

            self.saver = tf.train.Saver(tf.all_variables())
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            print('init')

            self.load_weights()
            #self.saver.restore(self.sess, "models/pose_gan_model-145")
            print ("load weights success")
            self.load_data()
            print ("load data success")

            loops= int(np.ceil(self.num_examples * 1.0 / self.batch_size / num_gpu))

            pix_lambda = 0.
            f_iter = 1

            #for i in range(self.epoch_num)
            #if i == 50:
            #    pix_lambda = 1.

            for epoch in range(0, self.epoch_num):
                np.random.shuffle(self.shuffle_ids)

                #f_loss = 0
                total_d_loss = 0.
                total_g_loss = 0.
                #total_f_loss = 0.
                if epoch == 50:
                    pix_lambda = 1.
                for i in range(loops):

                    feed = {}
                    for j in range(len(models)):
                        src_images, src_pose, src_app, trg_pose = self.get_batch()

                        feed[models[j][0]] = src_images
                        feed[models[j][1]] = src_pose
                        feed[models[j][2]] = trg_pose
                        feed[models[j][3]] = src_app
                        feed[models[j][4]] = pix_lambda

                    _, d_loss, g_loss = sess.run([all_op, aver_d_loss, aver_g_loss], feed)

                    total_d_loss += d_loss
                    total_g_loss += g_loss

                    now = time.time()
                    now = time.localtime(now)
                    now = time.strftime('%H-%M-%S', now)

                    print ('[%s]   epoch %d iter %d  d_loss %f   g_loss %f'%(now, epoch, i, d_loss, g_loss))

                    if i % 20 == 0:
                        scri, s2s_1, s2t_1, s2s_2, s2t_2 = sess.run([models[0][0], models[0][5], models[0][6], models[0][7], models[0][8]], feed)
                        scipy.misc.imsave("img/%d_%04d_0.jpg"%(epoch, i), np.concatenate([scri[0], s2s_2[0], s2t_2[0]], axis=1))
                        scipy.misc.imsave("img/%d_%04d_1.jpg"%(epoch, i), np.concatenate([scri[1], s2s_2[1], s2t_2[1]], axis=1))
                        scipy.misc.imsave("img/%d_%04d_2.jpg"%(epoch, i), np.concatenate([scri[2], s2s_2[2], s2t_2[2]], axis=1))
                        scipy.misc.imsave("img/%d_%04d_3.jpg"%(epoch, i), np.concatenate([scri[3], s2s_2[3], s2t_2[3]], axis=1))
                        scipy.misc.imsave("img/%d_%04d_small.jpg"%(epoch, i), np.concatenate([s2s_1[0], s2t_1[0]], axis=1))

                print('epoch %d g_loss %f    d_loss %f'%(epoch ,total_g_loss / loops, total_d_loss / loops))

                if epoch % 5 == 0 or epoch == 599:
                    self.save("models/pose_gan_model", epoch)

    def get_batch(self):

        current_ids = self.shuffle_ids[np.arange(self.start_idx, self.end_idx) % self.num_examples]
        self.start_idx += self.batch_size
        self.end_idx += self.batch_size
        sampled_images = self.images[current_ids] * (2. / 255) - 1

        pair_ids = [0] * self.batch_size
        for i in range(self.batch_size):
             pair_ids[i] = np.random.randint(self.num_examples)

        sampled_segs = self.segs[current_ids] * (2. / 255) - 1.
        pair_segs = self.segs[pair_ids] * (2. / 255) - 1.

        sampled_feature = self.googlenet[current_ids]

        transformed_images = np.zeros([sampled_images.shape[0], 256, 256, 3])
        transformed_mask = np.zeros([sampled_segs.shape[0], 64, 64, 3])
        transformed_pair_mask = np.zeros([sampled_segs.shape[0], 64, 64, 3])

        for i in range(self.batch_size):
            h = np.random.randint(0, 12)
            w = np.random.randint(0, 12)
            transformed_images[i] = sampled_images[i][w*4:w*4+256, h*4:h*4+256, :]
            transformed_mask[i] = sampled_segs[i][w:w+64, h:h+64, :]
            transformed_pair_mask[i] = pair_segs[i][w:w+64, h:h+64, :]

            if np.random.random() > 0.5:
                transformed_images[i] = np.fliplr(transformed_images[i])
                transformed_mask[i] = np.fliplr(transformed_mask[i])
                transformed_pair_mask[i] = np.fliplr(transformed_pair_mask[i])
            
	    #transformed_images[i] = cv2.resize(sampled_images[i], (256, 256))
	    #transformed_mask[i] = cv2.resize(sampled_segs[i], (64, 64))
	    #transformed_pair_mask[i] = cv2.resize(pair_segs[i], (64, 64))
	     		

        return transformed_images, transformed_mask, sampled_feature, transformed_pair_mask


    def save(self, name, step):
        self.saver.save(self.sess, name, global_step=step)

    def load(self):
        pass


if __name__ == '__main__':
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        args = {
            "batch_size": 24,
            "image_size_hr": 256,
            "image_size_lr": 64,
            "lr": 0.0002,
            "epoch_num": 600
        }
        model = PoseGAN(sess, args)
        model.train()
