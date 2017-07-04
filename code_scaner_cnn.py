# coding:utf-8
import tensorflow as tf
import numpy as np

characters = [chr(char) for char in range(48, 58)]
characters += [chr(char) for char in range(97, 123)]
characters += [chr(char) for char in range(65, 91)]

width, height, n_len, n_class = 72, 27, 4, len(characters)


def word2vec(words):
    result = {}
    for index, word in enumerate(words):
        result[word] = index
    return result


word2vec_dict = word2vec(characters)


def vec2word(vec):
    result = ''
    for index in vec:
        result += characters[index]
    return result


def convert_label(label):
    new_word = np.zeros([n_len, n_class])
    for index, word in enumerate(label):
        new_word[index][word2vec_dict[word]] = 1
    return new_word.reshape(-1, n_len * n_class)


def convert_image(image):
    image = image.convert('1')
    image = np.array(image).astype(np.float32)
    return image.reshape(-1, height * width)


def neural_net_image_input(image_shape):
    return tf.placeholder(tf.float32, (1, image_shape[0], image_shape[1], image_shape[2]), name='x')


def neural_net_label_input(label_shape):
    return tf.placeholder(tf.float32, (1, label_shape[0] * label_shape[1]), name='y')


def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, name='keep_prob')


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    weight = tf.Variable(
        tf.truncated_normal((list(conv_ksize) + [x_tensor.get_shape().as_list()[3], conv_num_outputs]), stddev=0.04))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    output = tf.nn.conv2d(x_tensor,
                          weight,
                          [1, conv_strides[0], conv_strides[1], 1],
                          padding='SAME')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.max_pool(output,
                            [1, pool_ksize[0], pool_ksize[1], 1],
                            [1, pool_strides[0], pool_strides[1], 1],
                            padding='SAME')
    return output


def flatten(x_tensor):
    shape = x_tensor.get_shape().as_list()
    return tf.reshape(x_tensor, [-1, shape[1] * shape[2] * shape[3]])


def fully_conn(x_tensor, num_outputs):
    weight = tf.Variable(tf.truncated_normal((x_tensor.get_shape().as_list()[1], num_outputs), stddev=0.04))
    bias = tf.Variable(tf.zeros(num_outputs))
    output = tf.add(tf.matmul(x_tensor, weight), bias)
    return tf.nn.relu(output)


def output(x_tensor, n_len, n_class):
    weight = tf.Variable(tf.truncated_normal((x_tensor.get_shape().as_list()[1], n_len * n_class), stddev=0.04))
    bias = tf.Variable(tf.zeros((n_len * n_class)))
    output = tf.add(tf.matmul(x_tensor, weight), bias)
    return output


def conv_net(x, keep_prob):
    conv = conv2d_maxpool(x,
                          conv_num_outputs=32,
                          conv_ksize=[3, 3],
                          conv_strides=[1, 1],
                          pool_ksize=[2, 2],
                          pool_strides=[2, 2])

    conv = conv2d_maxpool(conv,
                          conv_num_outputs=64,
                          conv_ksize=[3, 3],
                          conv_strides=[1, 1],
                          pool_ksize=[2, 2],
                          pool_strides=[2, 2])

    conv = conv2d_maxpool(conv,
                          conv_num_outputs=128,
                          conv_ksize=[3, 3],
                          conv_strides=[1, 1],
                          pool_ksize=[2, 2],
                          pool_strides=[2, 2])

    conv = conv2d_maxpool(conv,
                          conv_num_outputs=256,
                          conv_ksize=[3, 3],
                          conv_strides=[1, 1],
                          pool_ksize=[2, 2],
                          pool_strides=[2, 2])

    flt = flatten(conv)
    fc = fully_conn(flt, 1024)
    fc = tf.nn.dropout(fc, keep_prob)
    o = output(fc, n_len, n_class)
    return o


x = neural_net_image_input((height, width, 1))
keep_prob = neural_net_keep_prob_input()

logits = conv_net(x, keep_prob)
predict = tf.reshape(logits, [-1, n_len, n_class])
predict_text = tf.argmax(predict, 2)

save_model_path = './image_classification'
saver = tf.train.Saver()


def scaner(image):
    features = np.zeros((1, height * width))
    features[0, :] = convert_image(image)
    features = features.reshape(-1, height, width, 1)
    with tf.Session() as sess:
        saver.restore(sess, save_model_path)
        return vec2word(sess.run(predict_text, feed_dict={x: features, keep_prob: 1.})[0])
