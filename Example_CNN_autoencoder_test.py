import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

decoded_input = tf.placeholder(tf.float32, [None, 10],  name='input_code')
tf_is_training = tf.placeholder(tf.bool, None, name='training')
decoded_Dl1 = tf.layers.dense(decoded_input, 4*4*16, tf.nn.relu, name='Dense1')
decoded_image = tf.reshape(decoded_Dl1, [-1, 4, 4, 16])
# Decoder
upsample1 = tf.image.resize_images(decoded_image, size=(7, 7), method=tf.image.ResizeMethod.BILINEAR)
# Now 7x7x16
# w4 = weight_variable([3, 3, 16, 16])
# conv4 = tf.nn.conv2d_transpose(encoded, w4, output_shape=[bs, 7, 7, 16], strides=[1, 2, 2, 1], padding='SAME',
                               # name='conv4')
conv4 = tf.layers.conv2d_transpose(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same', strides=(1, 1),
                                   activation=tf.nn.relu, name='conv4')
# Now 7x7x16
upsample2 = tf.image.resize_images(conv4, size=(14, 14), method=tf.image.ResizeMethod.BILINEAR)
# Now 14x14x16
conv5 = tf.layers.conv2d_transpose(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1),
                                   activation=tf.nn.relu, name='conv5')

# Now 14x14x32
upsample3 = tf.image.resize_images(conv5, size=(28, 28), method=tf.image.ResizeMethod.BILINEAR)
# Now 28x28x32
conv6 = tf.layers.conv2d_transpose(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1),
                                   activation=tf.nn.relu, name='conv6')
# Now 28x28x32
# conv7 = tf.layers.conv2d_transpose(inputs=conv3, filters=32, kernel_size=(3, 3), strides=(2, 2),
# padding='same', activation=tf.nn.relu)
logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None, name='Dense2')
logits_normalized = tf.layers.batch_normalization(logits, training=tf_is_training, name='Bn1')
#Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits_normalized)
new_saver = tf.train.Saver()
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('D:\Tensorflow_Deep_Learning/autoencoder_net/autoencoder_net.ckpt.meta')
  new_saver.restore(sess, 'D:\Tensorflow_Deep_Learning/autoencoder_net/autoencoder_net.ckpt')
  # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
  decoded = tf.get_collection('decoded_result')[0]
  # graph = tf.get_default_graph()
  # input_x = graph.get_operation_by_name('input_code').outputs[0]
  # input_training = graph.get_operation_by_name('training').outputs[0]
  x_data = np.array([[0, 0, 0, 0.5, 0, 0, 0.5, 1, 0, 0]])
  result = sess.run(decoded, feed_dict={decoded_input: x_data, tf_is_training: False})
  # f, a = plt.subplots(1, 1, figsize=(1, 1))
  fig = plt.figure()
  ax = fig.add_subplot(121)
  ax.imshow(np.reshape(result[0], (28, 28)))
  plt.show()
  # for i in range(1):
      # a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
      #a[1][i].imshow(np.reshape(result[i], (28, 28)))
  # plt.show()