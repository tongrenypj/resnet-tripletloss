# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
import tensorflow as tf
import os
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS

## The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_string('version', 'train_56_batch256', '''A version number defining the directory to save
logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq', 2, '''Steps takes to output errors on the screen
and write summaries''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')


## The following flags define hyper-parameters regards training


tf.app.flags.DEFINE_float('margin', 1.0, '''margin''')

tf.app.flags.DEFINE_boolean('is_use_squared', True, '''Whether to use_squared''')
tf.app.flags.DEFINE_integer('train_steps', 8000, '''Total steps that you want to train''')
# tf.app.flags.DEFINE_boolean('is_full_validation', False, '''Validation w/ full validation set or
# a random batch''')
tf.app.flags.DEFINE_integer('train_batch_size', 256, '''Train batch size''')
# tf.app.flags.DEFINE_integer('validation_batch_size', 250, '''Validation batch size, better to be
# a divisor of 10000 for this task''')
tf.app.flags.DEFINE_integer('test_batch_size', 80, '''Test batch size''')

tf.app.flags.DEFINE_float('init_lr', 0.001, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the learning rate each
time''')
tf.app.flags.DEFINE_integer('decay_step0', 1500, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 3000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step2', 4500, '''At which step to decay the learning rate''')



## The following flags define hyper-parameters modifying the training network

tf.app.flags.DEFINE_integer('num_residual_blocks', 9, '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')


## The following flags are related to data-augmentation

# tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
# each side of the image''')


## If you want to load a checkpoint and continue training

# tf.app.flags.DEFINE_string('ckpt_path', 'cache/logs_repeat20/model.ckpt-100000', '''Checkpoint
# directory to restore''')

tf.app.flags.DEFINE_string('ckpt_path', 'logs_test_56_3batch256/model.ckpt-7800', '''Checkpoint
directory to restore''')
tf.app.flags.DEFINE_boolean('is_use_ckpt', True, '''Whether to load a checkpoint and continue
training''')

tf.app.flags.DEFINE_string('test_ckpt_path', 'logs_test_56_3batch256/model.ckpt-7800', '''Checkpoint
directory to restore''')


train_dir = 'logs_dir/' + FLAGS.version + '/'

NAME_TO_VISUALISE_VARIABLE = "cifar_embedding"
TO_EMBED_COUNT = 1200


path_for_cifar_sprites =  os.path.join(train_dir,'cifar20.png')
path_for_cifar_embedding_metadata =  os.path.join(train_dir,'metadata.tsv')


