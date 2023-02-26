import pandas as pd
import tensorflow as tf
import os

# import ants
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from mDCSRN.dataset import DatafromCSV
from mDCSRN.model import Generator


def score_patch(pred_patch, true_patch, c, CUBE=64):
    pred_patch = tf.reshape(pred_patch[0, ...], (-1, CUBE, CUBE, CUBE))
    true_patch = tf.reshape(true_patch[0, ...], (-1, CUBE, CUBE, CUBE))
    pred_patch = np.squeeze(pred_patch[:, :, :, :])
    true_patch = np.squeeze(true_patch[:, :, :, :])

    pred_cs = np.squeeze(pred_patch[c, :, :])
    pred_cs = tf.cast(tf.expand_dims(pred_cs, -1), dtype=tf.float64)
    true_cs = np.squeeze(true_patch[c, :, :])
    true_cs = tf.cast(tf.expand_dims(true_cs, -1), dtype=tf.float64)

    ssim = tf.image.ssim(pred_cs, true_cs, max_val=1.0)
    psnr = tf.image.psnr(pred_patch, true_patch, max_val=1.0)
    mse = tf.reduce_mean(tf.losses.mean_squared_error(pred_patch, true_patch))
    print('----------------------------------')
    print('SSIM:{} '.format(ssim.numpy()))
    print('PSNR:{} '.format(psnr.numpy()))
    print('MSE:{} '.format(mse.numpy()))
    print('----------------------------------')

# def registering(fix_path, mov_path):
#     # ants图片的读取
#     f_img = ants.image_read(fix_path)
#     m_img = ants.image_read(mov_path)
#     '''
#     ants.registration()函数的返回值是一个字典：
#         warpedmovout: 配准到fixed图像后的moving图像
#         warpedfixout: 配准到moving图像后的fixed图像
#         fwdtransforms: 从moving到fixed的形变场
#         invtransforms: 从fixed到moving的形变场
#
#     type_of_transform参数的取值可以为：
#         Rigid：刚体
#         Affine：仿射配准，即刚体+缩放
#         ElasticSyN：仿射配准+可变形配准，以MI为优化准则，以elastic为正则项
#         SyN：仿射配准+可变形配准，以MI为优化准则
#         SyNCC：仿射配准+可变形配准，以CC为优化准则
#     '''
#     # 图像配准
#     mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='SyN')
#     # 将形变场作用于moving图像，得到配准后的图像，interpolator也可以选择"nearestNeighbor"等
#     warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
#                                        interpolator="linear")
#
#     # 将配准后图像的direction/origin/spacing和原图保持一致
#     warped_img.set_direction(f_img.direction)
#     warped_img.set_origin(f_img.origin)
#     warped_img.set_spacing(f_img.spacing)
#     img_name = "./result/warped_img.nii.gz"
#
#     # 图像的保存
#     ants.image_write(warped_img, img_name)
#     print("End")


def disp(generator, sample_set, cube, step_counter):
    sample_set = sample_set.batch(2)
    PRED = np.array([], dtype=np.float32).reshape([0, cube[0], cube[1], cube[2], 1])
    HR = np.array([], dtype=np.float32).reshape([0, cube[0], cube[1], cube[2], 1])
    for idx, (lr, hr) in enumerate(sample_set):
        pred = np.array(generator.predict(lr.numpy()))
        pred = np.array(pred, dtype=np.float32)
        PRED = np.concatenate((PRED, pred))
        hr = np.array(hr, dtype=np.float32)
        HR = np.concatenate((HR, hr))
    inp = merge(PRED, cube, 1)
    tar = merge(HR, cube, 1)

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(inp[0, 18, :, :, :]))
    plt.title('predict')
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(tar[0, 18, :, :, :]))
    plt.title('truth')
    plt.axis('off')
    filename = './figure/' + str(step_counter) + '.png'
    Handle = plt.gcf()
    Handle.savefig(filename)
    plt.show()

    return inp, tar


def merge(concated_tensor, cube, x):
    print(concated_tensor.shape)
    d = int(34 / cube[0])
    d1 = int(624 / cube[1])
    d2 = int(543 / cube[2])
    d3 = int(cube[0] * cube[1] * cube[2])
    e = np.reshape(concated_tensor, [x, d, d1, d2, d3])
    f1 = np.split(e, x, axis=0)
    f2 = []
    for item in f1:
        f2.extend(np.split(item, d, axis=1))
    f3 = []
    for item in f2:
        f3.extend(np.split(item, d1, axis=2))
    f = []
    for item in f3:
        f.extend(np.split(item, d2, axis=3))
    g = [np.reshape(item, [1, cube[0], cube[1], cube[2], 1]) for item in f]
    h1 = [np.concatenate(g[d2 * i: d2 * (i + 1)], axis=3) for i in range(d * d1 * x)]
    h2 = [np.concatenate(h1[d1 * i: d1 * (i + 1)], axis=2) for i in range(d * x)]
    h3 = [np.concatenate(h2[d * i: d * (i + 1)], axis=1) for i in range(x)]
    h = np.concatenate(h3, axis=0)
    return h

def merge2(concated_tensor, cube, x, overlap=2):
    d = int(34 / cube[0])
    d1 = int(624 / cube[1])
    d2 = int(543 / cube[2])
    d3 = int(cube[0] * cube[1] * cube[2])
    e = np.reshape(concated_tensor, [x, d, d1, d2, d3])
    f1 = np.split(e, x, axis=0)
    f2 = []
    for item in f1:
        f2.extend(np.split(item, d, axis=1))
    f3 = []
    for item in f2:
        f3.extend(np.split(item, d1, axis=2))
    f = []
    for item in f3:
        f.extend(np.split(item, d2, axis=3))
    g = [np.reshape(item, [1, cube[0], cube[1], cube[2], 1]) for item in f]
    new_g = []
    for item in g:
        v = tf.squeeze(item)[:,overlap:cube[1]-overlap,overlap:cube[2]-overlap]
        new_g.append(tf.reshape(v, (1, 3, cube[1]-overlap*2, cube[2]-overlap*2, 1)))

    h1 = [np.concatenate(new_g[d2 * i: d2 * (i + 1)], axis=3) for i in range(d * d1 * x)]
    h2 = [np.concatenate(h1[d1 * i: d1 * (i + 1)], axis=2) for i in range(d * x)]
    h3 = [np.concatenate(h2[d * i: d * (i + 1)], axis=1) for i in range(x)]
    h = np.concatenate(h3, axis=0)
    return h


if __name__ == '__main__':
    x = tf.random.uniform([1, 3, 64, 64])
    y = tf.random.uniform([1, 3, 64, 64])
    score_patch(x, y, 25)


