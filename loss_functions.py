import tensorflow as tf


def supervised_loss(real, fake):
    mae = tf.keras.losses.MeanAbsoluteError()
    loss = mae(real, fake)  # L1 Loss
    # mse = tf.keras.losses.MeanSquaredError()
    # loss = mse(real, fake) # L2 Loss
    return loss

def celoss_ones(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1, ]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1, ]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, sr, hr):
    batch_sz = hr.shape[0]
    t = tf.random.uniform([batch_sz, 1, 1, 1, 1], dtype=tf.float64)
    t = tf.broadcast_to(t, hr.shape)
    hr = tf.cast(hr, dtype=tf.float64)
    #print(t.dtype, hr.dtype)
    interplate = t * hr + (1 - t) * sr

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplate_logits = discriminator(interplate)
    grads = tape.gradient(d_interplate_logits, interplate)

    # grads: [b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    gp = tf.reduce_mean((gp - 1) ** 2)
    gp = tf.cast(gp, dtype=tf.float64)
    return gp


def d_loss_fn(discriminator, sr, hr, training, lam=10.0):
    # 1. treat hr as real
    # 2. treat sr as fake
    d_fake_logits = discriminator(sr, training)
    loss = 0
    if training:
        loss = celoss_zeros(d_fake_logits)
    else:
        d_real_logits = discriminator(hr, training)
        d_fake_loss = celoss_zeros(d_fake_logits)
        d_real_loss = celoss_ones(d_real_logits)
    # gp = gradient_penalty(discriminator, sr, hr)
        loss = d_real_loss + d_fake_loss # + lam * gp
    loss = tf.cast(loss, dtype=tf.float64)
    return loss


def DLoss(y_pred, y_true):
  d_loss =  tf.reduce_mean(y_pred) - tf.reduce_mean(y_true)
  return d_loss