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


class PoseGAN:

    def __init__(self, sess, args):
        self.sess = sess
        self.image_size_hr = args["image_size_hr"]
        self.image_size_lr = args["image_size_lr"]

        with tf.variable_scope("app_encoder"):
            self.app_encoder = AppEncoder()

        with tf.variable_scope("generator"):
            self.generator = Generator()

        with tf.variable_scope("super_resolution"):
            self.super_resolution = SuperResolution()

        self.build()
        print ('init success')

    def build(self):
        self.src_images = tf.placeholder(tf.float32, [1, self.image_size_hr, self.image_size_hr, 3],
                        name = "source_images")

        self.trg_pose = tf.placeholder(tf.float32, [1, self.image_size_lr, self.image_size_lr, 3],
                        name = "target_pose")

        self.src_app = tf.placeholder(tf.float32, [1, 1024], name = "source_appearance")

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope('generator'):
                z = tf.random_normal(shape=(1, 100))
                with tf.variable_scope("g_net"):
                    self.fake_s2t_stageI = self.generator.build(self.src_app, z, self.trg_pose)

            with tf.variable_scope("super_resolution"):
                with tf.variable_scope("hr_g_net"):
                    self.fake_s2t_stageII = self.super_resolution.build(self.fake_s2t_stageI, self.src_app)

    def load_data(self):
        f = open('../final/Data/train/304images.pickle', 'rb')
        self.images = pickle.load(f)
        f.close()
        self.images = np.array(self.images)

        print('load image success')

        f = open('../final/Data/train/76segs.pickle', 'rb')
        self.segs = pickle.load(f)
        f.close()
        self.segs = np.array(self.segs)

        f = open('../final/Data/train/googlenet.pickle', 'rb')
        self.googlenet = pickle.load(f)
        f.close()


    def test(self):
        self.saver = tf.train.Saver(tf.all_variables())
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print('init')

        self.saver.restore(self.sess, "models/pose_gan_model-599")
        print ("load weights success")
        self.load_data()
        print ("load data success")

        #while True:
        #tmp = raw_input("input src_idx and trg_idx: [e.g.  101 202]        ")
        #src_idx, trg_idx = tmp.split()
        #src_idx = int(src_idx)
        #trg_idx = int(trg_idx)
        ids = [(7126, 7186), (7724, 7784)]
        #for i in range(2): 
            #src_idx = np.random.randint(2000)
            #trg_idx=  np.random.randint(2000)

	#ids = [2715, 2677, 2680, 2685, 2717, 2693, 2722, 2721, 2708,
	#       2725, 2718, 2727, 2732, 2689, 2713, 2714, 5942, 5975, 
	#       5993, 5997, 5941, 5949, 5963, 5962, 5944, 5961, 5951, 
	#       5984, 5989, 4675, 4660, 6883, 6881, 6879, 6827, 6877, 
	#       6861, 6857, 6848, 6927, 6931, 6933, 6892, 6515, 6524, 
	#       6511, 6002, 6058, 6006, 6021, 5999, 6010, 6009, 6023,
	#       6014, 6041, 6051, 6021, 6008, 6004, 6033, 6001, 6011,
        #       6056, 6048, 6016, 6958, 2005, 1979, 6554, 6544, 6581, 
	#       2975, 3028, 7051, 4689, 618, 6112, 6115, 6111, 6098, 
	#       7135, 7126, 7153, 6178, 6119, 6152, 6159, 6130, 6176,
	#       6158, 8834, 8842, 8832, 8434, 811, 824, 787, 6191, 
	#	6183, 6181, 6226, 6208, 6232, 6193, 1805, 1828, 2032, 
	#	2052, 2075, 6247, 2386, 2433, 5049, 506, 502, 508, 
	#	1319, 1357, 1362, 4610, 7321, 7326, 7358, 7350, 7335, 
	#	7450, 7465, 7477, 7459, 7520, 7543, 7498, 4099, 537,
	#	7663, 2105, 2140, 7675, 7691, 7692, 7669, 7768, 7733, 
	#	8256, 8202, 8332, 8374, 2792, 2796, 2813, 2812, 2811, 
	#	2825, 2804, 5298, 5866, 5921, 5915, 1520, 5595, 6782,
	#	6805, 6804, 8041, 8036]
	#ids = [787, 5595, 1520, 8832, 2708, 7350, 5921, 1805, 7691, 537,
	#	6934, 6892, 452, 453, 7321, 7326, 7327, 7464, 6788]
	#ids = [1805]

	#trg_idx = 1805
	for trg_idx in range(1786, 1845):
        	for src_idx in range(1786, 1845):
	
            		src_images, src_app, trg_images, trg_pose = self.get_images(src_idx, trg_idx)
            		feed = {
            		    self.src_images: src_images,
            		    self.src_app: src_app,
            		    self.trg_pose: trg_pose
            		}
            		res  = sess.run(self.fake_s2t_stageII, feed)
            		res_img = np.concatenate([src_images[0], trg_images[0],  res[0]], axis=1)
            		scipy.misc.imsave('output6/%d_%d.jpg'%(src_idx, trg_idx), res_img)
            		print('save to output6/%d_%d.jpg'%(src_idx, trg_idx))

    def get_images(self, src_idx, trg_idx):

        src_idx = [src_idx]
        trg_idx = [trg_idx]

        src_images = self.images[src_idx] * (2. / 255) - 1
        src_images = src_images[:, 24:280, 24:280, :]

        trg_images = self.images[trg_idx] * (2. / 255) - 1
        trg_images = trg_images[:, 24:280, 24:280, :]

        src_feature = self.googlenet[src_idx]

        trg_segs = self.segs[trg_idx] * (2. / 255) - 1
        trg_segs = trg_segs[:, 6:70, 6:70, :]

        return src_images, src_feature, trg_images, trg_segs


if __name__ == '__main__':
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        args = {
            "image_size_hr": 256,
            "image_size_lr": 64,
        }
        model = PoseGAN(sess, args)
        model.test()
