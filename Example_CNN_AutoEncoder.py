import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


learning_rate = 0.1
training_epochs = 100
batch_size = 50
display_step = 1
examples_to_show = 10

tf_is_training = tf.placeholder(tf.bool, None)
x = tf.placeholder(tf.float32, (None, 784), name='inputs')
x_image = tf.reshape(x, [-1, 28, 28, 1])
### Encoder
conv1 = tf.layers.conv2d(inputs=x_image, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
# Now 14x14x32
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
# Now 7x7x32
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
# Now 4x4x16
### Decoder
upsample1 = tf.image.resize_images(encoded, size=(7, 7), method=tf.image.ResizeMethod.BILINEAR)
# Now 7x7x16
conv4 = tf.layers.conv2d_transpose(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
upsample2 = tf.image.resize_images(conv4, size=(14, 14), method=tf.image.ResizeMethod.BILINEAR)
# Now 14x14x16
conv5 = tf.layers.conv2d_transpose(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
upsample3 = tf.image.resize_images(conv5, size=(28, 28), method=tf.image.ResizeMethod.BILINEAR)
# Now 28x28x32
conv6 = tf.layers.conv2d_transpose(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None)
logits_normalized = tf.layers.batch_normalization(logits, training=tf_is_training)
#Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits_normalized)
y = x_image
# Pass logits through sigmoid and calculate the cross-entropy loss
#loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=decoded)
loss = tf.pow(y - decoded, 2)
# # Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([opt, cost], feed_dict={x: batch_xs, tf_is_training: True})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

# # Applying encode and decode over test set
    encode_decode = sess.run(
        decoded, feed_dict={x: mnist.test.images[:examples_to_show], tf_is_training: False})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()
