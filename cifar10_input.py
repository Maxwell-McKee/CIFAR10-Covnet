import os
import tensorflow as tf 

NUM_TRAIN = 50000
NUM_TEST = 10000
IMAGE_DIM = 32

def read_dataset(file_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = 32*32*3
    record_bytes = label_bytes + image_bytes

    reader= tf.FixedLengthRecordReser(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes+image_bytes]),
                             [result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1,2,0])
    
    return result

def input(eval, path, batch_size):

    if eval_data:
        files = [os.path.join(path, 'test_batch.bin')]
        ex_per_epoch = NUM_TEST
    else:
        files = [os.path.join(path, 'data_batch_%d.bin' % i) for i in range(1,6)]
        ex_per_epoch = NUM_TRAIN

    for f in files:
        if not tf.gfile.Exists(f):
            raise ValueError('Could not find file: ' + f + '\nRe-download dataset or check directory.')

    with tf.name_scope('input'):
        file_queue = tf.train.string_intput_producer(files)

        read_input = read_dataset(file_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        float_image = tf.image.per_image_standardization(reshaped_image)

        float_image.set_shape([IMAGE_DIM, IMAGE_DIM, 3]) 
        read_input.label.set_shape([1])

        min_frac_ex_in_queue = 0.4
        min_examples = int(min_frac_ex_in_queue*ex_per_epoch)

        images, label_batch = tf.train.batch([float_image, read_input.label],
                                             batch_size,
                                             capacity=min_examples + 3*batch_size)
                                        
        return images, tf.reshape(label_batch, [batch_size])