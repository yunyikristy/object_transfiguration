import numpy as np 
import os
import pickle

data_dir = 'Data/birds'

label_file = os.path.join(data_dir, 'CUB_200_2011/image_class_labels.txt')
split_file = os.path.join(data_dir, 'CUB_200_2011/train_test_split.txt')
filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
train_filenames = os.path.join(data_dir, 'train/AppFilenames.pickle')
test_filenames = os.path.join(data_dir, 'test/AppFilenames.pickle')
train_labels = os.path.join(data_dir, 'train/App_class_info.pickle')
test_labels = os.path.join(data_dir, 'test/App_class_info.pickle')

f_labels = open(label_file, 'r')
f_split = open(split_file, 'r')
f_imgs = open(filepath, 'r')
f_train = open(train_filenames, 'w')
f_test = open(test_filenames, 'w')
f_train_labels = open(train_labels, 'w')
f_test_labels = open(test_labels, 'w')

train_name = []
test_name = []
train_labels = []
test_labels = []
for line, name, labels in zip(f_split, f_imgs, f_labels):
	indicator = int(line[-2])
	id = labels.split(' ')[1]
	label = int(id[:-1])
	# print('label is:', label)
	name = name[:-5].split(' ')[1]

	if indicator:		
		train_name.append(name)
		train_labels.append(label)
		# print('name and label is:', name, label)			
	else:
		test_name.append(name)
		test_labels.append(label)
		# print('name and label is:', name, label)


			
pickle.dump(train_name, f_train)
pickle.dump(test_name, f_test)
pickle.dump(train_labels, f_train_labels)
pickle.dump(test_labels, f_test_labels)

f_split.close()
f_imgs.close()
f_train.close()
f_test.close()
f_labels.close()
f_test_labels.close()
f_train_labels.close()
