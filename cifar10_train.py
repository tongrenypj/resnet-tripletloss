# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================

from resnet_inference import *
from datetime import datetime
import time
from input_image import *
import triplet_loss


class Res_Triplet_net(object):
    '''
    This Object is responsible for all the training and validation process
    '''

    def __init__(self):
        # Set up all the placeholders
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                       IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        # self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
        #                                                                       IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        # self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    def build_train_validation_graph(self):

        self.global_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        embeddings = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        # vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # loss = self.loss(logits, self.label_placeholder)

        self.loss, self.fraction_positive_triplets, self.num_positive_triplets, self.num_valid_triplets = \
            self.batch_all_triplet_loss(embeddings, self.label_placeholder, margin=FLAGS.margin,
                                        squared=FLAGS.is_use_squared)

        # [0.01]
        self.full_loss = tf.add_n([self.loss] + regu_losses)
        self.train_op, self.train_ema_op = self.train_operation(self.global_step, self.full_loss)

    def train(self):
        '''
        This is the main function for training
        '''

        # For the first step, we are loading all training images and validation images into the
        # memory

        # Build the graph for train and validation
        self.build_train_validation_graph()

        # 20 60 获取 batch_size数据

        img_batch, label_batch = get_batch_data(['tfrecord' + os.sep + 'cifar100_32train.tfrecords'],
                                                batch_size=FLAGS.train_batch_size)

        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        # saver = tf.train.Saver(tf.global_variables())
        saver = tf.train.Saver(max_to_keep=3)

        summary_op = tf.summary.merge_all()

        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)

        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print('Restored from checkpoint...')
            print('step start from %d...' % sess.run(self.global_step))

        start = sess.run(self.global_step)

        if start >= FLAGS.decay_step0:
            FLAGS.init_lr = 0.1 * FLAGS.init_lr
            print('Learning rate decayed to ', FLAGS.init_lr)

        if start >= FLAGS.decay_step1:
            FLAGS.init_lr = 0.1 * FLAGS.init_lr
            print('Learning rate decayed to ', FLAGS.init_lr)

        if start >= FLAGS.decay_step2:
            FLAGS.init_lr = 0.6 * FLAGS.init_lr
            print('Learning rate decayed to ', FLAGS.init_lr)

        print('Start training...')
        print('----------------------------')

        # 线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        # 非常重要读取数据线程

        for step in np.arange(start, FLAGS.train_steps):
            # Want to validate once before training. You may check the theoretical validation
            # loss first
            # 'report_freq', 391, '''Steps takes to output errors on the screen and write summaries

            # step = self.global_step
            # print(step, sess.run(self.global_step))

            batch_data, batch_label = sess.run([img_batch, label_batch])

            start_time = time.time()
            # operation.run()或tensor.eval()
            _, _, train_loss_value, real_loss, fraction_positive_triplets, \
            num_positive_triplets, num_valid_triplets = sess.run([self.train_op, self.train_ema_op,
                                                                  self.full_loss, self.loss,
                                                                  self.fraction_positive_triplets,
                                                                  self.num_positive_triplets,
                                                                  self.num_valid_triplets],
                                                                 feed_dict={self.image_placeholder: batch_data,
                                                                            self.label_placeholder: batch_label,
                                                                            self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time

            # 嵌入tensorboard可视化
            # ------------------------------------------------------------------------------------
            # embedding_var = tf.Variable(inference(all_img, FLAGS.num_residual_blocks, reuse=True),
            #                             name=NAME_TO_VISUALISE_VARIABLE)
            #
            # config = projector.ProjectorConfig()
            # embedding = config.embeddings.add()
            # embedding.tensor_name = embedding_var.name
            #
            # # Specify where you find the metadata
            # embedding.metadata_path = path_for_cifar_embedding_metadata  # 'metadata.tsv'
            #
            # # Specify where you find the sprite (we will create this later)
            # embedding.sprite.image_path = path_for_cifar_sprites  # 'mnistdigits.png'
            # embedding.sprite.single_image_dim.extend([32, 32])

            # ------------------------------------------------------------------------------------
            # Say that you want to visualise the embeddings
            # projector.visualize_embeddings(summary_writer, config)

            if step % FLAGS.report_freq == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: batch_data,
                                                    self.label_placeholder: batch_label,
                                                    self.lr_placeholder: FLAGS.init_lr})
                summary_writer.add_summary(summary_str, step)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: resnet--%d step %d, full_loss = %.6f '
                              'real_loss = %.6f (%.1f examples/sec; %.3f ' 'sec/batch)')

                print(format_str % (
                    datetime.now(), 6 * FLAGS.num_residual_blocks + 2, step, train_loss_value, real_loss,
                    examples_per_sec,
                    sec_per_batch))

                print('fraction_positive_triplets: %.6f ,num_positive_triplets: %d ,num_valid_triplets: %d' % (
                    fraction_positive_triplets, num_positive_triplets, num_valid_triplets))

                print('----------------------------')

            # if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
            #     FLAGS.init_lr = 0.1 * FLAGS.init_lr
            #     print('Learning rate decayed to ', FLAGS.init_lr)

            if step == FLAGS.decay_step0:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print('Learning rate decayed to ', FLAGS.init_lr)

            if step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print('Learning rate decayed to ', FLAGS.init_lr)

            if step == FLAGS.decay_step2:
                FLAGS.init_lr = 0.6 * FLAGS.init_lr
                print('Learning rate decayed to ', FLAGS.init_lr)
            # Save checkpoints every 10000 steps
            if step % 2 == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        sess.close()

    def test(self):

        # 20 60 获取 batch_size数据

        img_batch, label_batch = get_batch_data(['tfrecord' + os.sep + 'cifar100_32test.tfrecords'],
                                                        batch_size=FLAGS.test_batch_size)
        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        self.test_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.test_batch_size])

        # Build the test graph
        embeddings = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)

        result = self.batch_all_triplet_loss(embeddings, self.test_label_placeholder, margin=FLAGS.margin,
                                             squared=FLAGS.is_use_squared)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver()
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print('Model restored from ', FLAGS.test_ckpt_path)

        # 线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        # 非常重要读取数据线程

        print('Start testing...')
        print('----------------------------')
        for step in range(10):
            batch_data, batch_label = sess.run([img_batch, label_batch])

            start_time = time.time()
            # operation.run()或tensor.eval()

            # 0.34983072
            test_loss, fraction_positive_triplets, num_positive_triplets, num_valid_triplets = sess.run(
                result,
                feed_dict={self.test_image_placeholder: batch_data,
                           self.test_label_placeholder: batch_label
                           })
            duration = time.time() - start_time

            # print('fraction_positive_triplets: %.3f ,num_positive_triplets: %.3f ,num_valid_triplets: %.3f' % (
            #     fraction_positive_triplets, num_positive_triplets, num_valid_triplets))

            # print(
            #     fraction_positive_triplets, num_positive_triplets, num_valid_triplets)

            # print('test_loss：%.3f' % Test_loss)
            # print('----------------------------')
            # print('test_loss：%.6f fraction_positive_triplets: %.6f ,'
            #       'num_positive_triplets: %d ,num_valid_triplets: %d' % (test_loss, fraction_positive_triplets,
            #                                                              num_positive_triplets, num_valid_triplets))

            print(test_loss, fraction_positive_triplets,num_positive_triplets, num_valid_triplets)
            print('----------------------------')

        coord.request_stop()
        coord.join(threads)
        sess.close()

    def batch_all_triplet_loss(self, embeddings, labels, margin=0.5, squared=False):

        return triplet_loss.batch_all_triplet_loss(labels, embeddings, margin, squared=squared)

    def batch_hard_triplet_loss(self, embeddings, labels, margin=0.5, squared=False):

        return triplet_loss.batch_hard_triplet_loss(labels, embeddings, margin=margin, squared=squared)

    def train_operation(self, global_step, total_loss):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss])
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


# 发现复杂了可能效果还不好，所以就做了一个简单的模型，原始为32*32的图，padding了4位，然后再随机crop出32*32的图，
# 接着便三个卷积层结构，分别为32*32×16，16*16×32，8*8×64，每层n个block，每个block俩个卷积层，再加上最后fc共6n+2层。
# 说是110层效果最好，1000+层反而还不好了，可能是过拟合。

# Initialize the Train object
res_graph = Res_Triplet_net()
# Start the training session
# res_graph.train()
res_graph.test()
