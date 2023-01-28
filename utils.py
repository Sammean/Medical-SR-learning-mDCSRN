import numpy as np
import tensorflow as tf

def score_patch(pred_patch, true_patch, c):
    pred_patch = np.squeeze(pred_patch[0, :, :, :, :])
    true_patch = np.squeeze(true_patch[0, :, :, :, :])
    pred_cs = np.squeeze(pred_patch[c, :, :])
    pred_cs = tf.expand_dims(pred_cs, -1)
    true_cs = np.squeeze(true_patch[c, :, :])
    true_cs = tf.expand_dims(true_cs, -1)

    ssim = tf.image.ssim(pred_cs, true_cs, max_val=1.0)
    psnr = tf.image.psnr(pred_patch, true_patch, max_val=1.0)
    mse = tf.losses.mean_squared_error(pred_patch, true_patch)
    print('----------------------------------')
    print('SSIM:{} '.format(ssim.numpy()))
    print('PSNR:{} '.format(psnr.numpy()))
    print('MSE:{} '.format(mse.numpy()))
    print('----------------------------------')

if __name__ == '__main__':
    x = tf.random.uniform([2, 64, 64, 64, 1])
    y = tf.random.uniform([2, 64, 64, 64, 1])
    score_patch(x, y, 32)