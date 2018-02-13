import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
            tf.summary.histogram(layer_name+'/outputs', outputs)
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


x_data = np.linspace(-5, 5, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x_data.shape)
y_data = np.square(x_data) - 0.5


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # None means whatever number of samples is OK
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')  # must define the data type(there is no default)


l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.tanh)     # input data, input_size, output_size, activation function
l2 = add_layer(l1, 10, 10, n_layer=2, activation_function=tf.nn.sigmoid)
prediction = add_layer(l2, 10, 1, n_layer=3, activation_function=None)


with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data-prediction), reduction_indices=[1]))  # reduction_indices = [1] means compacts the matrix into a column, 0 for rows
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()  # Save the net
tf.add_to_collection('pred_network', prediction)
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("D:\Python Program", sess.graph)
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.show()
plt.ion()   # show and do not pause


for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})   # stochastic method uses part of data
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
        plt.cla()
        plt.scatter(x_data, y_data)
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        plt.plot(x_data, prediction_value, '-r', lw=5)
        plt.pause(0.1)
    if i == 999:
        save_path = saver.save(sess, "D:\Python Program/mynet/save_net.ckpt")  # Save the net
        print("Save to path", save_path)

plt.ioff()
plt.show()

# prediction_value = sess.run(prediction, feed_dict={xs: x_data})
# lines = ax.plot(x_data, prediction_value, '-r', lw=5)
# ax.lines.remove(lines[0])
# plt.pause
# print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
# try:
#     ax.lines.remove(lines[0])
# except Exception:
#     pass
# prediction_value = sess.run(prediction, feed_dict={xs: x_data})
# lines = ax.plot(x_data, prediction_value, '-r', lw=5)
# plt.pause(0.1)