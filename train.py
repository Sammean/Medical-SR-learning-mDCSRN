import os
import time

import tensorflow as tf
import numpy as np
from IPython import display

from mDCSRN.dataset import load_idset, DatafromCSV
from mDCSRN.model import Discriminator, Generator
from mDCSRN.loss_functions import supervised_loss, d_loss_fn
from mDCSRN.utils import score_patch

train_set, test_set, val_set, eval_set = load_idset('filelist.csv')
# ------------------ get one sample -------------------------------
# sample_id = val_set.take(1)
# sample_indices=np.array([idx.numpy() for a, idx in enumerate(sample_id)])
# Sampleloader = DatafromCSV(sample_indices)
# sample_set = Sampleloader.load_patchset()

# ========================== Formal Train =========================== #
# ------------------------ Hyper parameters ------------------- #
batch_size = 2
patch_size = 2
n_epochs = 10
BUFFER_SIZE = 777
n_patches = np.ceil(480/64)*np.ceil(672/64)*np.ceil(552/64)*batch_size

# -------------------------- Optimizer ----------------------- #

g_optimizer = tf.train.RMSPropOptimizer(0.0001)
d_optimizer = tf.train.RMSPropOptimizer(0.0001)
discriminator = Discriminator()
generator = Generator()
discriminator.compile(loss=d_loss_fn,
                      optimizer=d_optimizer,
                      metrics=['mse'])

# -------------------------- Checkpoint ----------------------- #
# Works On Colab
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_wgan")
checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer,
                                 d_optimizer=d_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# --------------- Marker for Loops Initialization ------------- #
batch_counter = 0
step_counter = 0  # control step for
d_loss_history = []
g_loss_history = []

flag_only_D = True  # flag for training Discriminator only, for the first 10k steps.
flag_G = False
# ================ Formal Training Process ================ #
'''
Basics: 
- Use pretrained generator: 
1) Train discriminator for 10k steps
2) D:G = 7:1, for 500 steps
3) D: extra 200 steps

2) + 3) for 550k steps totally

'''

for epoch in range(n_epochs):
    start = time.time()
    # ------------------- Split batch in idset ---------------------
    train_set = train_set.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size)
    iter = train_set.make_one_shot_iterator()
    for el in iter:
        batch_counter += 1
        display.clear_output(wait=True)
        indices = np.array(el)
        # ---------------------- Get data from id ----------------------
        Batchloader = DatafromCSV(indices)
        train_dataset = Batchloader.load_patchset()
        # -------------------- Split patch in dataset ------------------
        train_dataset = train_dataset.shuffle(buffer_size=n_patches).batch(patch_size)
        for patch, (lr, hr) in enumerate(train_dataset):
            step_counter += 1

            # train only discriminator for first 10k steps
            # from 10001 steps, reset step_counter and enter the formal loop
            flag_G = False
            if flag_only_D:
                if step_counter == 10001:
                    step_counter = 1
                    flag_only_D = False
                else:
                    print('Step for discriminator training:{}'.format(step_counter))

            if flag_only_D == False:
                if step_counter % 700 >= 500:
                    flag_G = False
                elif step_counter % 8 < 7:
                    flag_G = False
                else:
                    flag_G = True

                print('Global Step:{}, Subject No.{}, is training on G?: {}'.format(step_counter,
                                                                                    batch_counter,
                                                                                    flag_G))

            # training for generator
            if flag_G:
                with tf.GradientTape() as g_tape:

                    g_output = generator(lr, is_training=True)
                    d_fake_output = discriminator.predict(g_output.numpy())
                    g_vars = generator.variables

                    mse_loss = tf.losses.mean_squared_error(hr, g_output)  # L2 Loss for Generator(Parts)
                    g_gan_loss = - 1e-3 * tf.reduce_mean(d_fake_output)
                    g_loss = mse_loss + g_gan_loss
                    g_loss_history.append(g_loss)

                    with g_tape.stop_recording():
                        g_gradients = g_tape.gradient(g_loss, g_vars)  # generator.variables = g_vars
                        g_optimizer.apply_gradients(zip(g_gradients, g_vars))
                    print('Generator Loss:{}'.format(g_loss))
            else:
                if step_counter == 1 and flag_only_D:
                    temp = generator(tf.random.uniform([1, 64, 64, 64, 1]), is_training=False)
                    g_output = generator.predict(lr.numpy())
                    d_real_output = np.array([2, 1])
                else:
                    g_output = generator.predict(lr.numpy())
                    d_real_output = discriminator.predict(hr.numpy())

                d_loss, _ = discriminator.train_on_batch(g_output, d_real_output, reset_metrics=False)
                d_loss_history.append(d_loss)
                print('Discriminator Loss:{}'.format(d_loss))

            if step_counter % 200 == 0 and flag_only_D == False:
                manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=100)
                save_path = manager.save()
                print("\n----------Saved checkpoint for step {}: {}-------------\n".format(step_counter,
                                                                                           save_path))

                f = open('g_loss.txt', 'a')
                for g in g_loss_history:
                    f.write(str(g))
                    f.write('\n')
                f.close()
                g_loss_history = []

                f = open('h_loss.txt', 'a')
                for h in h_loss_history:
                    f.write(str(h))
                    f.write('\n')
                f.close()
                h_loss_history = []

            if step_counter % 301 == 0 and flag_only_D == False:
                # export evaluating parameters for [32,:,:] in a current patch
                score_patch(generator.predict(lr.numpy()), hr, 32)

            if step_counter % 701 == 0 and flag_only_D == False:
                display.clear_output(wait=True)

            if step_counter == 55000:
                print('\n ----------------- Completed for 55k steps! --------------------------\n')
                break
        if step_counter == 55000:
            break
    if step_counter == 55000:
        break
