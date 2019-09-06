from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


__all__ = ['loadData']

def datasetInit(dataset):
  file_names = {}
  if dataset=='cifar-10':
    FILENAME = 'cifar-10-python.tar.gz'
    DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + FILENAME
    LOCAL_FOLDER = 'cifar-10-batches-py'
    file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 6)]
    file_names['test'] = ['test_batch']
  elif dataset=='cifar-100':
    FILENAME= 'cifar-100-python.tar.gz'
    DOWNLOAD_URL='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    LOCAL_FOLDER = 'cifar-100-python'
    file_names['train'] = ['train']
    file_names['test'] = ['test']
  else:
    raise ValueError('`Dataset` must be one of `"CIFAR10"`, `"CIFAR100"`.')
  return FILENAME, DOWNLOAD_URL, LOCAL_FOLDER, file_names


def download_and_extract(data_dir, FILENAME, DOWNLOAD_URL):
  # download if not already downloaded.
  tf.contrib.learn.datasets.base.maybe_download(FILENAME, data_dir,
                                                DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, FILENAME),
               'r:gz').extractall(data_dir)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict


def convert_to_tfrecord(dataset, input_files, output_file):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      if dataset=='cifar-10':
        labels = data_dict[b'labels']
      elif dataset=='cifar-100':
        labels = data_dict[b'fine_labels']
      num_entries_in_batch = len(labels)
      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())


def loadData(dataset, data_dir, tfRecord=False): 
  FILENAME, DOWNLOAD_URL, LOCAL_FOLDER, file_names= datasetInit(dataset)
  print('Download from {} and extract.'.format(DOWNLOAD_URL))
  download_and_extract(data_dir, FILENAME, DOWNLOAD_URL)
  if tfRecord:    
    input_dir = os.path.join(data_dir, LOCAL_FOLDER)
    for mode, files in file_names.items():
      input_files = [os.path.join(input_dir, f) for f in files]
      output_file = os.path.join(data_dir, mode + '.tfrecords')
      try:
        os.remove(output_file)
      except OSError:
        pass
      # Convert to tf.train.Example and write the to TFRecords.
      convert_to_tfrecord(dataset, input_files, output_file)
  print('Done!')


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
                        "label": tf.FixedLenFeature([], tf.int64)}

    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    parsed_features['image'] = tf.decode_raw(
        parsed_features['image'], tf.uint8)

    return parsed_features['image'], parsed_features["label"]


def create_dataset(filepath, batch_size=128, shuffle_buffer=50000, num_classes=10):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.shuffle(shuffle_buffer).repeat()
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function)  # , num_parallel_calls=8)

    # Set the number of datapoints you want to load and shuffle

    # Set the batchsize
    dataset = dataset.batch(batch_size)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape
    image = tf.reshape(image, [-1, 32, 32, 3])

    # Create a one hot array for your labels
    label = tf.one_hot(label, num_classes)
    return image, label