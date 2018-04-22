from urllib.request import urlretrieve
from sklearn.preprocessing import LabelBinarizer
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
    return resized_img
    # return img[None, :, :, :]


def load_data(paths):
    imgs = {'low': [], 'mid': [], 'high': []}
    for k in imgs.keys():
        dir = paths + k
        for file in os.listdir(dir):
            if not file.lower().endswith('.png'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            imgs[k].append(resized_img)    # [1, height, width, depth] * n
            if len(imgs[k]) == 400:        # only use 400 imgs to reduce my memory load
                break
    # fake length data for tiger and cat
    low_y = np.zeros(len(imgs['low']), dtype="float32")
    mid_y = np.ones(len(imgs['mid']), dtype="float32")
    high_y = 2*np.ones(len(imgs['high']), dtype="float32")

    return imgs['low'], imgs['mid'], imgs['high'], low_y, mid_y, high_y


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters at here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, 3])
        self.tf_is_training = tf.placeholder(tf.bool, None)
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])
          # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_1 = tf.layers.dropout(conv1_1, rate=0.5, training=self.tf_is_training)
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_1 = tf.layers.dropout(conv2_1, rate=0.5, training=self.tf_is_training)
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_1 = tf.layers.dropout(conv3_1, rate=0.5, training=self.tf_is_training)
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_1 = tf.layers.dropout(conv4_1, rate=0.5, training=self.tf_is_training)
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_1 = tf.layers.dropout(conv5_1, rate=0.5, training=self.tf_is_training)
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 2048, tf.nn.relu, name='fc6')
        self.fc6 = tf.layers.dropout(self.fc6, rate=0, training=self.tf_is_training)
        self.fc6 = tf.layers.batch_normalization(self.fc6, training=self.tf_is_training)
        self.fc7 = tf.layers.dense(self.fc6, 1024, tf.nn.relu, name='fc7')
        # self.fc7 = tf.layers.dropout(self.fc7, rate=0.5, training=self.tf_is_training)
        self.fc7 = tf.layers.batch_normalization(self.fc7, training=self.tf_is_training)
        self.fc8 = tf.layers.dense(self.fc7, 512, tf.nn.sigmoid, name='fc8')
        self.fc9 = tf.layers.dense(self.fc8, 256, tf.nn.relu, name='fc9')
        self.fc9 = tf.layers.batch_normalization(self.fc9, training=self.tf_is_training)
        self.fc10 = tf.layers.dense(self.fc9, 64, tf.nn.relu, name='fc10')
        self.fc10 = tf.layers.batch_normalization(self.fc10, training=self.tf_is_training)
        self.fc11 = tf.layers.dense(self.fc10, 16, tf.nn.sigmoid, name='fc11')
        # self.fc8 = tf.layers.dropout(self.fc8, rate=0, training=self.tf_is_training)
        self.out = tf.layers.dense(self.fc11, 3, tf.nn.softmax, name='out')

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tfy, logits=self.out))
            # self.loss = tf.reduce_mean(-tf.reduce_sum(self.tfy * tf.log(self.out), reduction_indices=[1]))
            # self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.train_op = tf.train.AdamOptimizer(0.0005).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y, self.tf_is_training: True})
        return loss

    def predict(self, paths):
        low_tx, mid_tx, high_tx, low_ty, mid_ty, high_ty = load_data(paths)
        low_idx = np.random.randint(0, len(low_tx), 0)
        mid_idx = np.random.randint(0, len(mid_tx), 30)
        high_idx = np.random.randint(0, len(high_tx), 0)
        low_tx = np.array(low_tx)[low_idx]
        low_ty = np.array(low_ty)[low_idx]
        mid_tx = np.array(mid_tx)[mid_idx]
        mid_ty = np.array(mid_ty)[mid_idx]
        high_tx = np.array(high_tx)[high_idx]
        high_ty = np.array(high_ty)[high_idx]
        xt = np.concatenate(low_tx.tolist() + mid_tx.tolist() + high_tx.tolist(), axis=0) # axis=0)
        yt = np.concatenate((low_ty, mid_ty, high_ty), axis=0)
        yt = LabelBinarizer().fit_transform(yt)  # do not forget to plus ()
        total_size = len(xt)
        # Bar series
        y_pre = np.zeros([int(total_size), 3])
        for step in range(int(total_size)):
            y_pre[step, :] = self.sess.run(self.out, {self.tfx: xt[step: step + 1], self.tf_is_training: False})
        print("y_pre: ", y_pre.shape)
        correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(yt, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy is ', self.sess.run(accuracy))

    def save(self, path='D:\Research_Stuffs/Final Design/IMAGESET/DeepLearning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    low_x, mid_x, high_x, low_y, mid_y, high_y = load_data('D:\Research_Stuffs/Final Design/IMAGESET/DeepLearning/data_train/')

    # plot fake length distribution
    # plt.hist(tigers_y, bins=20, label='Tigers')
    # plt.hist(cats_y, bins=10, label='Cats')
    # plt.legend()
    # plt.xlabel('length')
    # plt.show()

    xs = np.concatenate(low_x + mid_x + high_x, axis=0)  # axis=0)
    ys = np.concatenate((low_y, mid_y, high_y), axis=0)
    ys = LabelBinarizer().fit_transform(ys)      # do not forget to plus ()

    vgg = Vgg16(vgg16_npy_path='D:\Research_Stuffs/Final Design/IMAGESET/DeepLearning/vgg16.npy')
    print('Net built')
    for i in range(5000):
        b_idx = np.random.randint(0, len(xs), 20)
        train_loss = vgg.train(xs[b_idx], ys[b_idx])
        print(i, 'train loss: ', train_loss)
        if (i+1) % 1000 == 0:
            vgg.save('D:\Research_Stuffs/Final Design/IMAGESET/DeepLearning/model/transfer_learn')

    vgg.save('D:\Research_Stuffs/Final Design/IMAGESET/DeepLearning/model/transfer_learn')      # save learned fc layers


def eval():
    vgg = Vgg16(vgg16_npy_path='D:\Research_Stuffs/Final Design/IMAGESET/DeepLearning/vgg16.npy',
                restore_from='D:\Research_Stuffs/Final Design/IMAGESET/DeepLearning/model/transfer_learn')
    vgg.predict('D:\Research_Stuffs/Final Design/IMAGESET/DeepLearning/data_test/')


if __name__ == '__main__':
    # download()
    # train()
    eval()