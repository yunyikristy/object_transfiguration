from __future__ import division
from __future__ import print_function


import numpy as np
import pickle
import random


class Dataset(object):
    def __init__(self, images, imsize, conditions, embeddings=None, 
                 filenames=None, workdir=None,
                 labels=None, aug_flag=True,
                 class_id=None, class_range=None):
        self._images = images
        self._conditions = conditions
        self._embeddings = embeddings
        self._filenames = filenames
        self.workdir = workdir
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = len(images)
        self._saveIDs = self.saveIDs()

        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._aug_flag = aug_flag
        self._class_id = np.array(class_id)
        self._class_range = class_range
        self._imsize = imsize
        self._perm = None

    @property
    def images(self):
        return self._images

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def filenames(self):
        return self._filenames

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def saveIDs(self):
        self._saveIDs = np.arange(self._num_examples)
        np.random.shuffle(self._saveIDs)
        return self._saveIDs
    
    def transform(self, images):
        if self._aug_flag:
            transformed_images =\
                np.zeros([images.shape[0], self._imsize, self._imsize, 5])
            ori_size = images.shape[1]
            for i in range(images.shape[0]):
                h1 = int(np.floor((ori_size - self._imsize) * np.random.random()))
                w1 = int(np.floor((ori_size - self._imsize) * np.random.random()))
                cropped_image =\
                    images[i][w1: w1 + self._imsize, h1: h1 + self._imsize, :]
                transformed_images[i] = cropped_image
            return transformed_images
        else:
            return images

    def next_batch(self, batch_size, window):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self._perm = np.arange(self._num_examples)
            np.random.shuffle(self._perm)

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        current_ids = self._perm[start:end]
        fake_ids = np.random.randint(self._num_examples, size=batch_size)
        collision_flag =\
            (self._class_id[current_ids] == self._class_id[fake_ids])
        fake_ids[collision_flag] =\
            (fake_ids[collision_flag] +
             np.random.randint(100, 200)) % self._num_examples

        sampled_images = self._images[current_ids]
        sampled_wrong_images = self._images[fake_ids, :, :, :]
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_wrong_images = sampled_wrong_images * (2. / 255) - 1.
        
        #sampled_images = self.transform(sampled_images)
        #sampled_wrong_images = self.transform(sampled_wrong_images)
        ret_list = [sampled_images, sampled_wrong_images]

        if self._conditions is not None:
            sampled_conditions = self._conditions[current_ids]
            sampled_wrong_conditions = self._conditions[fake_ids, :, :, :]
            sampled_conditions = sampled_conditions.astype(np.float32)
            sampled_wrong_conditions = sampled_wrong_conditions.astype(np.float32)
            #sampled_conditions = sampled_conditions * (2. / 255) - 1.
            #sampled_wrong_conditions = sampled_wrong_conditions * (2. / 255) - 1.

            #sampled_conditions = self.transform(sampled_conditions)
            #sampled_wrong_conditions = self.transform(sampled_wrong_conditions)
            ret_list.append(sampled_conditions)
        else:
            ret_list.append(None)

        if self._labels is not None:
            ret_list.append(self._labels[current_ids])
        else:
            ret_list.append(None)
        return ret_list

    def next_batch_test(self, batch_size, start):
        """Return the next `batch_size` examples from this data set."""
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
            start = end - batch_size
        else:
            end = start + batch_size

        sampled_images = self._images[start:end]
        sampled_images = sampled_images.astype(np.float32)
        # from [0, 255] to [-1.0, 1.0]
        sampled_images = sampled_images * (2. / 255) - 1.
        # sampled_images = self.transform(sampled_images)

        sampled_conditions = self._conditions[start:end]
        sampled_conditions = sampled_conditions.astype(np.float32)
        # from [0, 255] to [-1.0, 1.0]
        # sampled_conditions = sampled_conditions * (2. / 255) - 1.
        # sampled_conditions = self.transform(sampled_conditions)

        return [sampled_images, sampled_conditions, 
                self._saveIDs[start:end]]


class TextDataset(object):
    def __init__(self, workdir, hr_lr_ratio):
        lr_imsize = 64
        self.hr_lr_ratio = hr_lr_ratio
        if self.hr_lr_ratio == 1:
            self.image_filename = '/76segs.pickle'
            self.condition_filename = '/76parts.pickle'
        elif self.hr_lr_ratio == 4:
            self.image_filename = '/304segs.pickle'
            self.condition_filename = '/304parts.pickle'

        #self.image_shape = [lr_imsize * self.hr_lr_ratio,
        #                    lr_imsize * self.hr_lr_ratio, 3]
        self.image_shape = [76, 76, 3]
       
        self.train = None
        self.test = None
        self.workdir = workdir

    def get_data(self, pickle_path, aug_flag=True):
        with open(pickle_path + self.image_filename, 'rb') as f:
            images = pickle.load(f)
            images = np.array(images)
            images = images[:, :, :, :1]
            print(images.shape)
            print('images: ', images.shape)
            

        with open(pickle_path + self.condition_filename, 'rb') as f:
            conditions_pickle = pickle.load(f)
            conditions = np.zeros((len(conditions_pickle), 76, 76, 5), dtype='uint8')
            for i in range(len(conditions)):
                coord = conditions_pickle[i]
                for j in range(5):
                    minx = coord[j][0]
                    maxx = coord[j][1]
                    miny = coord[j][2]
                    maxy = coord[j][3]
                    conditions[i, miny:maxy, minx:maxx, j] = 1
            #conditions = np.array(conditions)
            print('conditions: ', conditions.shape)

        with open(pickle_path + '/filenames.pickle', 'rb') as f:
            list_filenames = pickle.load(f)
            print('list_filenames: ', len(list_filenames), list_filenames[0])
        with open(pickle_path + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f)
            print('class_id:', len(class_id))
        return Dataset(images, self.image_shape[0], conditions,
                       filenames=list_filenames, workdir=self.workdir, labels=None,
                       aug_flag=aug_flag, class_id=class_id)
# def __init__(self, images, imsize, conditions, embeddings=None, 
#                 filenames=None, workdir=None,
#                 labels=None, aug_flag=True,
#                 class_id=None, class_range=None):

