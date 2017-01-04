import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv_weights(name, shape):
    W = tf.Variable(tf.zeros(list(shape)), name=name + "kernel")
    b = tf.Variable(tf.zeros([shape[-1]]), name=name + "biaises")
    return W, b


def full_weights(name, shape):
    W = tf.Variable(tf.zeros(list(shape)), name=name + "biaises")
    b = tf.Variable(tf.zeros([shape[-1]]), name=name + "biaises")
    return W, b


def convolutional_layer(inp, kernel, biases, stride, pad, name):
    pad = [[pad, pad]] * 2
    temp = tf.pad(inp, [[0, 0]] + pad + [[0, 0]])
    temp = tf.nn.conv2d(temp, kernel, padding='VALID',
                        name=name, strides=[1] + [stride] * 2 + [1])
    return tf.nn.bias_add(temp, biases)


def max_pool(inp, ksize, stride, name):
    return tf.nn.max_pool(
        inp, padding='SAME',
        ksize=[1] + [ksize] * 2 + [1],
        strides=[1] + [stride] * 2 + [1],
        name=name)


def fully_connected(inp, weights, biases, name):
    return tf.nn.xw_plus_b(inp, weights, biases, name)


def create_graph(inp_size):
    all_vars = []

    inp = tf.placeholder(tf.float32, inp_size, 'input')

    feed = dict()  # other placeholders
    # Load  |  Yep!  | scale to (-1, 1)                 | (?, 448, 448, 3)
    temp = inp * 2. - 1.
    # Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 448, 448, 16)
    kernel, biases = conv_weights("1-convolutional/", (3, 3, 3, 16))
    all_vars.append(kernel)
    all_vars.append(biases)
    temp = convolutional_layer(temp, kernel, biases, 1, 1, "1_convolutionnal")
    temp = tf.maximum(.1 * temp, temp, name="2_leaky")

    # Load  |  Yep!  | maxp 2x2p0_2                     | (?, 224, 224, 16)
    temp = max_pool(temp, 2, 2, "3_maxpool")
    # Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 224, 224, 32)
    kernel, biases = conv_weights("4-convolutional/", (3, 3, 16, 32))
    all_vars.append(kernel)
    all_vars.append(biases)
    temp = convolutional_layer(temp, kernel, biases, 1, 1, "4-convolutionnal")
    temp = tf.maximum(.1 * temp, temp, name="5_leaky")
    # Load  |  Yep!  | maxp 2x2p0_2                     | (?, 112, 112, 32)
    temp = max_pool(temp, 2, 2, "6_maxpool")
    # Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 112, 112, 64)
    kernel, biases = conv_weights("7-convolutional/", (3, 3, 32, 64))
    all_vars.append(kernel)
    all_vars.append(biases)
    temp = convolutional_layer(temp, kernel, biases, 1, 1, "7-convolutionnal")
    temp = tf.maximum(.1 * temp, temp, name="8_leaky")
    # Load  |  Yep!  | maxp 2x2p0_2                     | (?, 56, 56, 64)
    temp = max_pool(temp, 2, 2, "9_maxpool")
    # Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 56, 56, 128)
    kernel, biases = conv_weights("10-convolutional/", (3, 3, 64, 128))
    all_vars.append(kernel)
    all_vars.append(biases)
    temp = convolutional_layer(temp, kernel, biases, 1, 1, "10-convolutionnal")
    temp = tf.maximum(.1 * temp, temp, name="11_leaky")
    # Load  |  Yep!  | maxp 2x2p0_2                     | (?, 28, 28, 128)
    temp = max_pool(temp, 2, 2, "12_maxpool")
    # Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 28, 28, 256)
    kernel, biases = conv_weights("13-convolutional/", (3, 3, 128, 256))
    all_vars.append(kernel)
    all_vars.append(biases)
    temp = convolutional_layer(temp, kernel, biases, 1, 1, "13-convolutionnal")
    temp = tf.maximum(.1 * temp, temp, name="14_leaky")
    # Load  |  Yep!  | maxp 2x2p0_2                     | (?, 14, 14, 256)
    temp = max_pool(temp, 2, 2, "15_maxpool")
    # Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 14, 14, 512)
    kernel, biases = conv_weights("16-convolutional/", (3, 3, 256, 512))
    all_vars.append(kernel)
    all_vars.append(biases)
    temp = convolutional_layer(temp, kernel, biases, 1, 1, "16-convolutionnal")
    temp = tf.maximum(.1 * temp, temp, name="17_leaky")
    # Load  |  Yep!  | maxp 2x2p0_2                     | (?, 7, 7, 512)
    temp = max_pool(temp, 2, 2, "18_maxpool")
    # Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
    kernel, biases = conv_weights("19-convolutional/", (3, 3, 512, 1024))
    all_vars.append(kernel)
    all_vars.append(biases)
    temp = convolutional_layer(temp, kernel, biases, 1, 1, "19-convolutionnal")
    temp = tf.maximum(.1 * temp, temp, name="20_leaky")
    # Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
    kernel, biases = conv_weights("21-convolutional/", (3, 3, 1024, 1024))
    all_vars.append(kernel)
    all_vars.append(biases)
    temp = convolutional_layer(temp, kernel, biases, 1, 1, "21-convolutionnal")
    temp = tf.maximum(.1 * temp, temp, name="22_leaky")
    # Init  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
    kernel, biases = conv_weights("23-convolutional/", (3, 3, 1024, 1024))
    all_vars.append(kernel)
    all_vars.append(biases)
    temp = convolutional_layer(temp, kernel, biases, 1, 1, "22-convolutionnal")
    temp = tf.maximum(.1 * temp, temp, name="23_leaky")
    # Load  |  Yep!  | flat                             | (?, 50176)
    temp = tf.transpose(temp, [0, 3, 1, 2])
    temp = slim.flatten(temp, scope="24_flat")

    weights, biases = full_weights("26-connected/", (50176, 256))
    all_vars.append(weights)
    all_vars.append(biases)
    # Init  |  Yep!  | full 50176 x 256  linear         | (?, 256)
    temp = fully_connected(temp, weights, biases, "25_fully_connected")
    weights, biases = full_weights("27-connected/", (256, 4096))
    all_vars.append(weights)
    all_vars.append(biases)
    # Init  |  Yep!  | full 256 x 4096  leaky           | (?, 4096)
    temp = fully_connected(temp, weights, biases, "27_fully_connected")
    temp = tf.maximum(.1 * temp, temp, name="27_leaky")

    # Load  |  Yep!  | drop                             | (?, 4096)
    # dropout
    keep_prob = tf.placeholder(tf.float32)
    temp = tf.nn.dropout(temp, keep_prob, name="28_drop")
    # Init  |  Yep!  | full 4096 x 1470  linear         | (?, 1470)

    # TRANSFER LEARNING HERE TODO find out why 5 classes 735 and 12 -> 1078
    weights = tf.Variable(tf.truncated_normal([4096, 1078], stddev=0.1),
                          name="30-connected/weights")
    biases = tf.Variable(tf.constant(0.1, shape=[1078]),
                         name="30-connected/biases")
    all_vars.append(weights)
    all_vars.append(biases)
    temp = fully_connected(temp, weights, biases, "29_fully_connected")

    return temp, all_vars, [inp, keep_prob]


def locate_layer(inp, pad, kernel):
    pad = [[pad, pad]] * 2
    temp = tf.pad(inp, [[0, 0]] + pad + [[0, 0]])

    ksz = self.lay.ksize
    half = ksz/2
    out = list()
    for i in range(self.lay.h_out):
        row_i = list()
        for j in range(self.lay.w_out):
            kij = kernel[i * self.lay.w_out + j]
            i_, j_ = i + 1 - half, j + 1 - half
            tij = temp[:, i_: i_ + ksz, j_: j_ + ksz, :]
            row_i.append(tf.nn.conv2d(tij, kij, padding='VALID',
                                      strides=[1] * 4))
            out += [tf.concat(2, row_i)]

    return tf.concat(1, out)
