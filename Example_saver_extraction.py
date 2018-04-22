# Use the model trained in the "Example_Activation_Function"
import tensorflow as tf
import numpy as np
# 在下面的代码中，默认加载了TensorFlow计算图上定义的全部变量
# 直接加载持久化的图
# 如果不希望重新定义图上的运算，也可以直接加载已经持久化的图
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('D:\Python Program/mynet/save_net.ckpt.meta')
  new_saver.restore(sess, 'D:\Python Program/mynet/save_net.ckpt')
  # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
  y = tf.get_collection('pred_network')[0]

  graph = tf.get_default_graph()

  # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
  input_x = graph.get_operation_by_name('inputs/x_input').outputs[0] # Name with name scope
  x_data = np.linspace(-5, 5, 11)[:, np.newaxis]

  # 使用y进行预测
  print(x_data)
  print("Now is y_data")
  result_y = sess.run(y, feed_dict={input_x: x_data})
  print(result_y)