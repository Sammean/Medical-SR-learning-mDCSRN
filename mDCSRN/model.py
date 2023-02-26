import tensorflow as tf
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import layers, Sequential
from tensorflow.keras import backend as K


k = 16
filter_size = 3
NUMBER_OF_UNITS_PER_BLOCK = 4
utilize_bias = False
w_init = tf.keras.initializers.GlorotUniform()
# w_init = tf.keras.initializers.HeUniform()


def dense_unit(no_of_filters=k, f_size=filter_size):
    unit = Sequential([
        layers.BatchNormalization(),
        layers.ELU(),
        layers.Conv3D(no_of_filters, f_size, kernel_initializer=w_init, use_bias=utilize_bias,
                      padding='same', dtype=tf.float64)
    ])
    return unit

def dense_block(inputs, num_units=NUMBER_OF_UNITS_PER_BLOCK, training=True):

    for i in range(num_units):
        dense_unit_output = dense_unit(k, filter_size, training=training)(inputs)
        inputs = tf.keras.layers.Concatenate(dtype=tf.float64)([inputs, dense_unit_output])

    return inputs


def Generator(patch_size=[64, 64, 64], training=True):
    patch_size.append(1)
    inputs = tf.keras.layers.Input(shape=patch_size, dtype=tf.float64)
    conv0 = layers.Conv3D(2 * k, filter_size, kernel_initializer=w_init, use_bias=utilize_bias,
                          padding='same', dtype=tf.float64)(inputs)

    dense0 = dense_block(conv0, training=training)
    concat = layers.Concatenate(dtype=tf.float64)([conv0, dense0])
    compress0 = layers.Conv3D(2 * k, 1, padding='same', dtype=tf.float64)(concat)

    dense1 = dense_block(compress0, training=training)
    concat = layers.Concatenate(dtype=tf.float64)([concat, dense1])
    compress1 = layers.Conv3D(2 * k, 1, padding='same', dtype=tf.float64)(concat)

    dense2 = dense_block(compress1, training=training)
    concat = layers.Concatenate(dtype=tf.float64)([concat, dense2])
    compress2 = layers.Conv3D(2 * k, 1, padding='same', dtype=tf.float64)(concat)

    dense3 = dense_block(compress2, training=training)
    concat = layers.Concatenate(dtype=tf.float64)([concat, dense3])

    reconstruction = layers.Conv3D(1, 1, padding='same', dtype=tf.float64)(concat)
    return tf.keras.Model(inputs=inputs, outputs=reconstruction, name='generator')



def conv_stride_block(no_of_filters):
    block = Sequential([
        layers.Conv3D(no_of_filters, 3, padding='same', strides=1,
                      dtype=tf.float64, kernel_constraint=WeightClip(0.01)),
        layers.LayerNormalization(),
        layers.LeakyReLU(),

        layers.Conv3D(no_of_filters, 3, padding='same', strides=2,
                      dtype=tf.float64, kernel_constraint=WeightClip(0.01)),
        layers.LayerNormalization(),
        layers.LeakyReLU()
    ])

    return block


# ============== Weights Clip ============= #
class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

def Discriminator(patch_size=[64, 64, 64], no_filters=64):
    patch_size.append(1)
    inputs = tf.keras.layers.Input(shape=patch_size, dtype=tf.float64)

    conv1 = layers.Conv3D(no_filters, 3, strides=1, padding='same',
                          dtype=tf.float64, activation=layers.LeakyReLU(),
                          kernel_constraint=WeightClip(0.01))(inputs)

    conv2 = layers.Conv3D(no_filters, 3, strides=2, padding='same',
                          dtype=tf.float64, kernel_constraint=WeightClip(0.01))(conv1)
    lnorm = layers.LayerNormalization()(conv2)
    lrelu = layers.LeakyReLU()(lnorm)

    csb1_out = conv_stride_block(no_filters * 2)(lrelu)
    csb2_out = conv_stride_block(no_filters * 4)(csb1_out)
    csb3_out = conv_stride_block(no_filters * 8)(csb2_out)

    flatten = layers.Flatten()(csb3_out)
    fc = layers.Dense(1024, activation=layers.LeakyReLU(), kernel_constraint=WeightClip(0.01))(flatten)
    logits = layers.Dense(1, kernel_constraint=WeightClip(0.01))(fc)


    return tf.keras.Model(inputs=inputs, outputs=logits, name='discriminator')



if __name__ == '__main__':
    x = tf.random.normal([1, 3, 64, 64, 1, 1])
    discriminator = Discriminator()
    d_optimizer = tf.optimizers.Adam()
    checkpoint_dir = './tensor_checkpoints_d'
    ckpt_d = tf.train.Checkpoint(discriminator=discriminator,
                                 d_optimizer=d_optimizer)
    ckpt_manager_d = tf.train.CheckpointManager(ckpt_d, checkpoint_dir, max_to_keep=10)
    #if ckpt_manager_d.latest_checkpoint:


    g = Generator(patch_size=[3,64,64,1])
    g.summary()
    out = g(x)
    print(out.shape)

    # d = Discriminator(64,64)
    # d.summary()
    # logits = d(x)
    # print(logits)
