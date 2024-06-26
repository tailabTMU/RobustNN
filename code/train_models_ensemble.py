## original code by Nicholas Carlini unless otherwise indicated

## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from aggregation import noisy_max

import tensorflow as tf
from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR
import os


def train(train_data, train_labels, validation_data, validation_labels,file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    print(train_data.shape)
    
    model.add(Conv2D(params[0], (3, 3),
                            input_shape=train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(10))
    
    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(train_data, train_labels,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              nb_epoch=num_epochs,
              shuffle=True)
    

    if file_name != None:
        model.save(file_name)

    return model


## train all teacher models based on original train function by Nicolas Carlini
## use partitioning of training data for MNIST, no partitioning for CIFAR
def train_teachers_all(dataset, params, arrTemp, train_fileprefix, arrInit=None, num_epochs=50,id_start=0, bool_partition=True):

  nb_teachers = len(arrTemp)
  for teacher_id in range(id_start, nb_teachers):
    # Retrieve subset of data for this teacher
    if bool_partition:
      train_data, train_labels = partition_dataset(dataset.train_data,
                                            dataset.train_labels,
                                            nb_teachers,
                                            teacher_id)
    else:
      train_data, train_labels = dataset.train_data, dataset.train_labels

    print("Length of training data: " + str(len(train_labels)))

    # Define teacher checkpoint filename and full path
    filename = train_fileprefix+str(nb_teachers) + '_teachers_' + str(teacher_id)
    init = None
    if arrInit != None:
      init = arrInit[teacher_id]
    teacher_model = train(train_data, train_labels, dataset.validation_data, dataset.validation_labels,filename,params,num_epochs=num_epochs,train_temp=arrTemp[teacher_id],init=init)

  return True

## partition function for training data
def partition_dataset(data, labels, nb_teachers, teacher_id):

  # Sanity check
  assert len(data) == len(labels)
  assert int(teacher_id) < int(nb_teachers)

  # This will floor the possible number of batches
  batch_len = int(len(data) / nb_teachers)

  # Compute start, end indices of partition
  start = teacher_id * batch_len
  end = (teacher_id+1) * batch_len

  # Slice partition off
  partition_data = data[start:end]
  partition_labels = labels[start:end]

  return partition_data, partition_labels

def check_accuracy(model, test_data, test_labels, image_size, num_channels, batch_size=1):
  with tf.Session() as sess:
    x = tf.placeholder(tf.float32, (None, image_size, image_size, num_channels))
    y = model.predict(x)
    r = []
    for i in range(0,len(test_data),batch_size):
        pred = sess.run(y, {x: test_data[i:i+batch_size]})
        r.append(np.argmax(pred,1) == np.argmax(test_labels[i:i+batch_size],1))
        print(np.mean(r))
    return np.mean(r)
