#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10

classes1 = ['car', 'bird', 'plane']
# 0 ,1
classes = ['cat', 'dog']

cifar_100_classes = np.arange(20).astype('str')


# 制作二进制数据
def convert_to_tf_record(classes_list, path=None, tf_record_name='tfrecord' + os.sep + 'train.tfrecord'):
    if path is None:
        path = os.getcwd()
    writer = tf.python_io.TFRecordWriter(tf_record_name)
    for index, label_name in enumerate(classes_list):

        class_path = path + os.sep + label_name + os.sep
        # class_path = os.path.join(cwd,label_name)

        for img_name in os.listdir(class_path):
            if img_name.endswith('.png') or img_name.endswith('.jpg'):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((32, 32))
                img_raw = img.tobytes()  # 将图片转化为原生bytes
                # print(label_name, index, img_raw)
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))
                writer.write(example.SerializeToString())
        print(index, label_name, img_raw)
    writer.close()


# convert_to_tf_record(cifar_100_classes)


# 读取二进制数据
def read_and_decode(filename, num_epochs=None, shuffle=True):
    # 文件队列 ，文件阅读器 ，文件阅读器的解析器 ，启动文件队列
    # 创建文件队列,不限读取的数量，来生成一个先入先出的队列， 文件阅读器会需要它来读取数据。
    filename_queue = tf.train.string_input_producer(filename, num_epochs=num_epochs, shuffle=shuffle)
    # create a reader from file queue，根据你的文件格式， 选择对应的文件阅读器， 然后将文件名队列提供给阅读器的read方法
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    keys, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本  ，从TFRecords文件中读取数据， 可以使用tf.TFRecordReader的tf.parse_single_example解析器
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [32, 32, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label


def get_shuffle_batch_data(filename, batch_size):
    image, label = read_and_decode(filename, num_epochs=None, shuffle=False)

    # min_after_dequeue = 1000
    # capacity = min_after_dequeue + 3 * 4

    min_after_dequeue = 700
    capacity = 1000
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                      capacity=capacity, min_after_dequeue=min_after_dequeue,
                                                      num_threads=2, )
    return image_batch, label_batch


def get_batch_data(filename, batch_size):
    image, label = read_and_decode(filename, num_epochs=None, shuffle=False)

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=2,
                                              capacity=1000)

    return image_batch, label_batch


if __name__ == '__main__':

    ok = False

    if ok:
        data = convert_to_tf_record(classes)
    else:
        # img,label = read_and_decode(['train.tfrecords','train.tfrecords1'])
        # img = read_and_decode("train.tfrecords", None, True)

        img, label = read_and_decode(['cifar100_32train.tfrecords'], num_epochs=None, shuffle=False)

        print("tengxing", img, label)

        # 使用shuffle_batch可以随机打乱输入 next_batch挨着往下取
        # shuffle_batch才能实现[img,label]的同步,也即特征和label的同步,不然可能输入的特征和label不匹配
        # 比如只有这样使用,才能使img和label一一对应,每次提取一个image和对应的label
        # shuffle_batch返回的值就是RandomShuffleQueue.dequeue_many()的结果
        # Shuffle_batch构建了一个RandomShuffleQueue，并不断地把单个的[img,label],送入队列中
        # img_batch, label_batch = tf.train.shuffle_batch([img, label],
        #                                             batch_size=3, capacity=2000,
        #                                             min_after_dequeue=1000)

        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * 4

        img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=12,
                                                        num_threads=3,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)

        # img_batch, label_batch = tf.train.batch([img, label], batch_size=3,
        #                                         num_threads=3,
        #                                         capacity=1000,
        #                                         )
        # 初始化所有的op

        with tf.Session() as sess:
            init = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            # 启动队列，这个QueueRunner的工作线程是独立于文件阅读器的线程， 因此乱序和将文件名推入到文件名队列这些过程不会阻塞文件阅读器运行。
            # 你必须调用tf.train.start_queue_runners来将文件名填充到队列，否则read操作会被阻塞到文件名队列中有值为止。
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            # operation.run()或tensor.eval()
            step = 1
            for i in range(step):
                image_batch_v, label_batch_v = sess.run([img_batch, label_batch])
                print(image_batch_v.shape, label_batch_v)
                for img in image_batch_v:
                    print(img.shape)
                    # print(img)
                    plt.imshow(img)
                    plt.show()

            coord.request_stop()
            coord.join(threads)
