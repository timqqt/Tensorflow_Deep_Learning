import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# tf.set_random_seed(2)
# np.random.seed(2)

# Hyper Parameters
BATCH_SIZE = 256
LR_G = 0.0001          # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 10             # think of this as number of ideas for generating an art work (Generator)
examples_to_show = 10
# ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
# PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
tf_is_training = tf.placeholder(tf.bool, None, name='sign_of_training')
real_art = tf.placeholder(tf.float32, [None, 784], name='real_in')
real_art_image = tf.reshape(real_art, [-1, 28, 28, 1])
ideas = tf.placeholder(tf.float32, [None, N_IDEAS], name='ideals_in')


def artist_works():     # painting from the famous artist (real target)
    paintings, paintings_labels = mnist.train.next_batch(BATCH_SIZE)
    return paintings, paintings_labels


def generator(input):
    with tf.variable_scope('Generator'):
        G_in = input       # random ideas (could from normal distribution)
        G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
        G_l2 = tf.layers.dense(G_l1, 256, tf.nn.relu)
        # G_l2 = tf.layers.batch_normalization(G_l2, training=tf_is_training)
        # G_l3 = tf.layers.dense(G_l2, 784, tf.nn.sigmoid)
        G_l2 = tf.reshape(G_l2, [-1, 4, 4, 16])
        upsample1 = tf.image.resize_images(G_l2, size=(7, 7), method=tf.image.ResizeMethod.BICUBIC)
        # Now 7x7x16
        conv4 = tf.layers.conv2d_transpose(inputs=upsample1, filters=16, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.relu)
        # Now 7x7x16
        upsample2 = tf.image.resize_images(conv4, size=(14, 14), method=tf.image.ResizeMethod.BICUBIC)
        # Now 14x14x16
        conv5 = tf.layers.conv2d_transpose(inputs=upsample2, filters=32, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.relu)
        # Now 14x14x32
        upsample3 = tf.image.resize_images(conv5, size=(28, 28), method=tf.image.ResizeMethod.BICUBIC)
        # Now 28x28x32
        conv6 = tf.layers.conv2d_transpose(inputs=upsample3, filters=32, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.relu)
        # Now 28x28x32
        logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None)
        logits_normalized = tf.layers.batch_normalization(logits, training=tf_is_training)
        # Now 28x28x1
        # Pass logits through sigmoid to get reconstructed image
        # G_out = tf.nn.sigmoid(logits_normalized)
        G_out = tf.reshape(logits_normalized, [-1, 28, 28, 1])
                     # making a painting from these random ideas
    return G_out


def discriminator(input, reuse=False):
    with tf.variable_scope('Discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        feed_art = input   # receive art work from the famous artist
        # D_in = tf.reshape(feed_art, [-1, 784])
        ### Encoder
        conv1 = tf.layers.conv2d(inputs=feed_art, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # # Now feed_art
        maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
        # # Now 14x14x32
        conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # # Now 14x14x32
        maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 7x7x32
        conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
        # Now 7x7x16
        conv_out = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')
        # Now 4x4x16
        conv_flatten = tf.reshape(conv_out, [-1, 4*4*16])
        D_l0 = tf.layers.dense(conv_flatten, 256, tf.nn.relu)
        D_l1 = tf.layers.dense(D_l0, 128, tf.nn.relu)
        D_l2 = tf.layers.dense(D_l1, 64, tf.nn.relu)
        D_normalized = tf.layers.batch_normalization(D_l2, training=tf_is_training)
        prob_art = tf.layers.dense(D_normalized, 1, tf.nn.sigmoid, name='prob')
        return prob_art


prob_artist0 = discriminator(real_art_image, reuse=False)
fake_art = generator(ideas)
prob_artist1 = discriminator(fake_art, reuse=True)
D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1-prob_artist1))
G_loss = tf.reduce_mean(tf.log(1-prob_artist1))

train_D = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   # something about continuous plotting
for step in range(5000):
    artist_paintings, G_labels = artist_works()           # real painting from artist
    G_ideas = np.random.normal(0, 1, size=(BATCH_SIZE, N_IDEAS)).astype(np.float32)
    # Train Discriminator
    loss_discriminator = sess.run([D_loss, train_D], feed_dict={ideas: G_ideas,
                                                                               tf_is_training: True,
                                                                               real_art: artist_paintings
                                                                              })[:1]
    if step % 1 == 0:
        loss_generator = sess.run([G_loss, train_G], feed_dict={ideas: G_ideas, tf_is_training: True, real_art: artist_paintings})[:1]
    #print("Step now: ", step+1)
    if step % 100 == 0 or step == 4999:
        print("Loss of Generator is ", loss_generator, "\n")
        print("Loss of Discriminator is ", loss_discriminator, "\n")
        # # Applying encode and decode over test set
        test_ideas = np.random.normal(0, 1, size=(examples_to_show, N_IDEAS)).astype(np.float32)
        generate_art = sess.run(
            fake_art, feed_dict={ideas: test_ideas, tf_is_training: False})
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(generate_art[i], (28, 28)))
        plt.show()
    # train generator
    # G_paintings, pa0, pa1, Dl = sess.run([G_out, prob_artist0, prob_artist1, D_loss, train_D, train_G],    # train discriminator
    #                                 {G_in: G_ideas, feed_art: artist_paintings})[:4]

print("Training Finished!")

