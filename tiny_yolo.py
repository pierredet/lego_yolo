import tensorflow as tf
import tensorflow.contrib.slim as slim


inp_size = [None] + self.meta['inp_size']
inp = tf.placeholder(tf.float32, inp_size, 'input')

feed = dict()  # other placeholders


def conv_weights(name, shape):
    W = tf.Variable(tf.zeros(list(shape)), name=name+"kernel")
    b = tf.Variable(tf.zeros([shape[-1]]), name=name+"biaises")
    return W, b


def full_weights(name, shape):
    W = tf.Variable(tf.zeros(list(shape)), name=name+"biaises")
    b = tf.Variable(tf.zeros([shape[-1]]), name=name+"biaises")
    return W, b


full_weights("26-connected/", (50176, 256))
full_weights("27-connected/", (256, 4096))
full_weights("30-connected/", (4096, 1470))


def convolutional_layer(inp, kernel, biases, stride, pad, name):
    pad = [[pad, pad]] * 2
    temp = tf.pad(inp, [[0, 0]] + pad + [[0, 0]])
    temp = tf.nn.conv2d(temp, kernel, padding='VALID',
                        name=name, strides=[1] + [stride] * 2 + [1])
    return tf.nn.bias_add(temp, biases)


def max_pool(inp, ksize, stride, name):
    return = tf.nn.max_pool(
        inp, padding='SAME',
        ksize=[1] + [ksize] * 2 + [1],
        strides=[1] + [stride] * 2 + [1],
        name=name)


def fully_connected(inp, weights, biases, name):
    inp = tf.nn.xw_plus_b(inp, weights, biases,	name)

# Load  |  Yep!  | scale to (-1, 1)                 | (?, 448, 448, 3)
temp = inp * 2. - 1.
# Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 448, 448, 16)
kernel, biases = conv_weights("1-convolutional/", (3, 3, 3, 16))
temp = convolutional_layer(temp, kernel, biases, 1, 1, "1_convolutionnal")
temp = tf.maximum(.1 * temp, temp, name="2_leaky")

# Load  |  Yep!  | maxp 2x2p0_2                     | (?, 224, 224, 16)
temp = max_pool(temp, 2, 0, "3_maxpool")
# Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 224, 224, 32)
kernel, biases = conv_weights("4-convolutional/", (3, 3, 16, 32))
temp = convolutional_layer(temp, kernel, biases, 1, 1, "4-convolutionnal")
temp = tf.maximum(.1 * temp, temp, name="5_leaky")
# Load  |  Yep!  | maxp 2x2p0_2                     | (?, 112, 112, 32)
temp = max_pool(temp, 2, 0, "6_maxpool")
# Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 112, 112, 64)
kernel, biases = conv_weights("7-convolutional/", (3, 3, 32, 64))
temp = convolutional_layer(temp, kernel, biases, 1, 1, "7-convolutionnal")
temp = tf.maximum(.1 * temp, temp, name="8_leaky")
# Load  |  Yep!  | maxp 2x2p0_2                     | (?, 56, 56, 64)
temp = max_pool(temp, 2, 0, "9_maxpool")
# Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 56, 56, 128)
kernel, biases = conv_weights("10-convolutional/", (3, 3, 64, 128))
temp = convolutional_layer(temp, kernel, biases, 1, 1, "10-convolutionnal")
temp = tf.maximum(.1 * temp, temp, name="11_leaky")
# Load  |  Yep!  | maxp 2x2p0_2                     | (?, 28, 28, 128)
temp = max_pool(temp, 2, 0, "12_maxpool")
# Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 28, 28, 256)
kernel, biases = conv_weights("13-convolutional/", (3, 3, 128, 256))
temp = convolutional_layer(temp, kernel, biases, 1, 1, "13-convolutionnal")
temp = tf.maximum(.1 * temp, temp, name="14_leaky")
# Load  |  Yep!  | maxp 2x2p0_2                     | (?, 14, 14, 256)
temp = max_pool(temp, 2, 0, "15_maxpool")
# Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 14, 14, 512)
kernel, biases = conv_weights("16-convolutional/", (3, 3, 256, 512))
temp = convolutional_layer(temp, kernel, biases, 1, 1, "16-convolutionnal")
temp = tf.maximum(.1 * temp, temp, name="17_leaky")
# Load  |  Yep!  | maxp 2x2p0_2                     | (?, 7, 7, 512)
temp = max_pool(temp, 2, 0, "18_maxpool")
# Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
kernel, biases = conv_weights("19-convolutional/", (3, 3, 512, 1024))
temp = convolutional_layer(temp, kernel, biases, 1, 1, "19-convolutionnal")
temp = tf.maximum(.1 * temp, temp, name="20_leaky")
# Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
kernel, biases = conv_weights("21-convolutional/", (3, 3, 1024, 1024))
temp = convolutional_layer(temp, kernel, biases, 1, 1, "21-convolutionnal")
temp = tf.maximum(.1 * temp, temp, name="22_leaky")
# Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
kernel, biases = conv_weights("23-convolutional/", (3, 3, 1024, 1024))
temp = convolutional_layer(temp, kernel, biases, 1, 1, "22-convolutionnal")
temp = tf.maximum(.1 * temp, temp, name="23_leaky")
# Load  |  Yep!  | flat                             | (?, 50176)
temp = tf.transpose(temp, [0, 3, 1, 2])
temp = slim.flatten(temp, scope="24_flat")

weights, biases = full_weights("26-connected/", (50176, 256))
# Init  |  Yep!  | full 50176 x 256  linear         | (?, 256)
temp = fully_connected(temp, weights, biases, "25_fully_connected")
weights, biases = full_weights("27-connected/", (256, 4096))
# Init  |  Yep!  | full 256 x 4096  leaky           | (?, 4096)
temp = fully_connected(temp, weights, biases, "27_fully_connected")
temp = tf.maximum(.1 * temp, temp, name="27_leaky")

# Load  |  Yep!  | drop                             | (?, 4096)
temp = tf.nn.dropout(temp, self.lay.h['pdrop'], name="28_drop")
# Init  |  Yep!  | full 4096 x 1470  linear         | (?, 1470)

weights, biases = full_weights("30-connected/", (4096, 1470))
temp = fully_connected(temp, weights, biases, "29_fully_connected")
