from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import sys
import time
from datetime import datetime

from xdsj_detection.yolo.solver.solver import Solver

class YoloSolver(Solver):
  """Yolo Solver 
  """
  def __init__(self, dataset, net, common_params, solver_params):
    #process params
    self.moment = float(solver_params['moment'])
    self.learning_rate = float(solver_params['learning_rate'])
    self.batch_size = int(common_params['batch_size'])
    self.height = int(common_params['image_size'])
    self.width = int(common_params['image_size'])
    self.max_objects = int(common_params['max_objects_per_image'])
    self.pretrain_path = str(solver_params['pretrain_model_path'])
    self.train_dir = str(solver_params['train_dir'])
    self.max_iterators = int(solver_params['max_iterators'])
    #
    self.dataset = dataset
    self.net = net
    #construct graph
    self.construct_graph()

  def _train(self):
    """Train model

    Create an optimizer and apply to all trainable variables.

    Args:
      total_loss: Total loss from net.loss()
      global_step: Integer Variable counting the number of training steps
      processed
    Returns:
      train_op: op for training
    """
    # global_step = tf.Variable(0)
    # learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=self.max_iterators / self.batch_size, decay_rate=0.98,
    #                                            staircase=True)

    """
    训练分为两个阶段，第一阶段里前面又划分出一段作为“热身阶段”：
    热身阶段：learn_rate = (global_step / warmup_steps) * learn_rate_init
    其他阶段：learn_rate_end + 0.5 * (learn_rate_init - learn_rate_end) * (
             1 + tf.cos((global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
    
    with tf.name_scope('learn_rate'):
      self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
      warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                 dtype=tf.float64, name='warmup_steps')  # warmup_periods epochs
      train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                dtype=tf.float64, name='train_steps')
      self.learn_rate = tf.cond(
        pred=self.global_step < warmup_steps,
        true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
        false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) * (
          1 + tf.cos((self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi)))
      global_step_update = tf.assign_add(self.global_step, 1.0)
    """

    # opt = tf.train.MomentumOptimizer(self.learning_rate, self.moment)
    opt = tf.train.AdamOptimizer(self.learning_rate,1e-08)
    grads = opt.compute_gradients(self.total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

    return apply_gradient_op

  def construct_graph(self):
    # construct graph
    self.global_step = tf.Variable(0, trainable=False)
    self.images = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3))
    self.labels = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
    self.objects_num = tf.placeholder(tf.int32, (self.batch_size))

    self.predicts = self.net.inference(self.images)

    self.total_loss, self.nilboy = self.net.loss(self.predicts, self.labels, self.objects_num)
    
    tf.summary.scalar('loss', self.total_loss)
    self.train_op = self._train()

  def solve(self):
    saver1 = tf.train.Saver(self.net.pretrained_collection, write_version=1)
    saver2 = tf.train.Saver(self.net.trainable_collection, write_version=1)

    init =  tf.global_variables_initializer()

    #结果保存，以便tensorboard显示
    summary_op = tf.summary.merge_all()

    #开启会话
    sess = tf.Session()

    #初始化
    sess.run(init)
    print(self.pretrain_path)
    saver1.restore(sess, self.pretrain_path)

    #保存日志
    summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

    #迭代训练
    for step in range(self.max_iterators):
      #计时
      start_time = time.time()
      np_images, np_labels, np_objects_num = self.dataset.batch()

      #开始训练
      _, loss_value, nilboy = sess.run([self.train_op, self.total_loss, self.nilboy], feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})
      # print("%%%%%%%%%%%%%%%%%%%")
      # print(loss_value, nilboy)
      # print("%%%%%%%%%%%%%%%%%%%")

      #结束计时
      duration = time.time() - start_time

      #异常判断
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      #每10次循环，打印结果
      if step % 10 == 0:
        num_examples_per_step = self.dataset.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

        # format_str = ('%s: step %d, loss = %.2f, train_acc = %.2f (%.1f examples/sec; %.3f '
        #               'sec/batch)')
        # print (format_str % (datetime.now(), step, loss_value, nilboy,
        #                      examples_per_sec, sec_per_batch))

        sys.stdout.flush()
      #每100次更新日志
      if step % 100 == 0:
        summary_str = sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels, self.objects_num: np_objects_num})
        summary_writer.add_summary(summary_str, step)
      #每5000次保存模型
      if step % 5000 == 0:
        print("save model:",self.train_dir + '/model.ckpt')
        saver2.save(sess, self.train_dir + '/model.ckpt', global_step=step)
    #关闭会话
    sess.close()
