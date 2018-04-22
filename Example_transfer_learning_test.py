# Use the model trained in the "Example_Activation_Function"
import tensorflow as tf
import numpy as np
# 在下面的代码中，默认加载了TensorFlow计算图上定义的全部变量
# 直接加载持久化的图
# 如果不希望重新定义图上的运算，也可以直接加载已经持久化的图

x_data = np.linspace(-5, 5, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x_data.shape)
y_data = np.power(x_data, 2) - 0.5

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('D:\Python Program/mynet/save_net.ckpt.meta')
  # new_saver.restore(sess, 'D:\Python Program/mynet/save_net.ckpt')
  # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
  y = tf.get_collection('pred_network')[0]
  graph = tf.get_default_graph()
  # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
  input_x = graph.get_operation_by_name('inputs/x_input').outputs[0]  # Name with name scope
  xs = y
  ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
  l1 = tf.layers.dense(xs, 10, tf.nn.relu, name='l')
  l2 = tf.layers.dense(l1, 10, tf.nn.relu, name='2')
  prediction = tf.layers.dense(l2, 1, activation=None, name='pred')
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction), reduction_indices=[1]))
  train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
  init = tf.global_variables_initializer()
  sess.run(init)
  for i in range(10000):
      result = sess.run([loss, prediction, train_step], feed_dict={input_x: x_data, ys: y_data})[0]  # stochastic method uses part of data
      if i % 50 == 0:
          print(result)
