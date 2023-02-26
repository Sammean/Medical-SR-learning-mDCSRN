import os
import time

import tensorflow as tf
import numpy as np
from IPython import display

from mDCSRN.dataset import load_idset, DatafromCSV
from mDCSRN.model import Discriminator, Generator
from mDCSRN.loss_functions import supervised_loss, d_loss_fn, DLoss, gradient_penalty
from mDCSRN.utils import score_patch
from mDCSRN.logger import log


def pretrain_loop(csv_path, batch_size, patch_size, n_epochs, learning_rate):
    train_set, test_set, val_set, eval_set = load_idset(csv_path)
    # ------------------ get one sample -------------------------------
    sample_id = val_set.take(1)
    sample_indices = np.array([idx.numpy() for a, idx in enumerate(sample_id)])
    Sampleloader = DatafromCSV(sample_indices)
    sample_set = Sampleloader.load_patchset()

    # ========================== Initial Train =========================== #
    # ----- The Loss is need further validation: from valid images --------#
    # ------------------------ Hyper parameters ------------------- #
    CUBE = 16
    BUFFER_SIZE = 777
    n_patches = np.ceil(34 / CUBE) * np.ceil(624 / CUBE) * np.ceil(523 / CUBE) * batch_size
    beta1 = 0.5

    # -------------------------- Optimizer ----------------------- #
    g_optimizer_init = tf.keras.optimizers.Adam(learning_rate, beta1=beta1)

    generator = Generator(patch_size=CUBE)
    generator.compile(loss=supervised_loss,
                      optimizer=g_optimizer_init,
                      metrics=['mse'])

    # -------------------------- Checkpoint ----------------------- #
    # Works On Colab
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(g_optimizer_init=g_optimizer_init,
                                     generator=generator)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # --------------- Marker for Loops Initialization ------------- #
    batch_counter = 0
    step_counter = 0
    g_loss_history = []
    valid_loss_history = []

    for epoch in range(n_epochs):
        start = time.time()
        # ------------------- Split batch in idset ---------------------
        train_set = train_set.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size)
        iter = tf.compat.v1.data.make_one_shot_iterator(train_set)
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
                print('Global Step:{}, Subject No.{}'.format(step_counter, batch_counter))
                g_loss, _ = generator.train_on_batch(lr, hr, reset_metrics=False)
                g_loss_history.append(g_loss)

                if step_counter % 200 == 0:
                    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)
                    save_path = manager.save()
                    print("\n----------Saved checkpoint for step {}: {}-------------\n".format(step_counter, save_path))
                if step_counter == 50000:
                    print('\n ----------------- Completed for 50k steps! --------------------------\n')
                    break
            if step_counter == 50000:
                break
        if step_counter == 50000:
            break

def train_loop(csv_path, batch_size, patch_size, n_epochs):
    train_set, test_set, val_set, eval_set = load_idset(csv_path)
    # ------------------ get one sample -------------------------------
    # sample_id = val_set.take(1)
    # sample_indices=np.array([idx.numpy() for a, idx in enumerate(sample_id)])
    # Sampleloader = DatafromCSV(sample_indices)
    # sample_set = Sampleloader.load_patchset()

    # ========================== Formal Train =========================== #
    # ------------------------ Hyper parameters ------------------- #
    # batch_size = 2
    # patch_size = 2
    # n_epochs = 10
    CUBE = 16
    BUFFER_SIZE = 777
    n_patches = np.ceil(34/CUBE)*np.ceil(624/CUBE)*np.ceil(523/CUBE)*batch_size

    # -------------------------- Optimizer ----------------------- #

    g_optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0001)
    d_optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0001)
    discriminator = Discriminator(patch_size=CUBE)
    generator = Generator(patch_size=CUBE)
    discriminator.compile(loss=DLoss,
                          optimizer=d_optimizer,
                          metrics=['mse'])

    # -------------------------- Checkpoint ----------------------- #
    path = "./training_checkpoints"
    ckpt_g = tf.train.Checkpoint(generator_g=generator,
                                 generator_optimizer=g_optimizer)

    ckpt_manager_g = tf.train.CheckpointManager(ckpt_g, path, max_to_keep=10)
    if ckpt_manager_g.latest_checkpoint:
        ckpt_g.restore(ckpt_manager_g.latest_checkpoint)
        log('Generator: Latest checkpoint restored!!')
    else:
        log("Generator: No checkpoint found! Staring from scratch!")

    checkpoint_dir = './training_checkpoints_d'
    ckpt_d = tf.train.Checkpoint(discriminator=discriminator,
                                 d_optimizer=d_optimizer)
    ckpt_manager_d = tf.train.CheckpointManager(ckpt_d, checkpoint_dir, max_to_keep=10)
    if ckpt_manager_d.latest_checkpoint:
        ckpt_d.restore(ckpt_manager_d.latest_checkpoint)
        log('Discriminator: Latest checkpoint restored!!')
    else:
        log("Discriminator: No checkpoint found! Staring from scratch!")

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
        iter = tf.compat.v1.data.make_one_shot_iterator(train_set)
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

                        g_output = generator(lr, training=True)
                        #d_fake_output = discriminator(g_output.numpy())
                        g_vars = generator.variables

                        mae_loss = tf.losses.mean_absolute_error(hr, g_output)  # L1 Loss for Generator(Parts)
                        #g_gan_loss = tf.cast(- 1e-3 * tf.reduce_mean(d_fake_output), dtype=tf.float64)
                        gp = gradient_penalty(discriminator, g_output, hr)
                        g_loss = tf.reduce_mean(mae_loss) + gp
                        g_loss_history.append(g_loss)

                        with g_tape.stop_recording():
                            g_gradients = g_tape.gradient(g_loss, g_vars)  # generator.variables = g_vars
                            g_optimizer.apply_gradients(zip(g_gradients, g_vars))
                        print('Generator Loss:{}'.format(g_loss))
                else:
                    with tf.GradientTape() as d_tape:

                        if step_counter == 1 and flag_only_D:
                            # temp = generator(tf.random.uniform([1, CUBE, CUBE, CUBE, 1]), itraining=False)
                            g_output = generator(lr.numpy())
                            d_real_output = np.zeros([2.0, 1.0])
                        else:
                            g_output = generator(lr.numpy())
                            d_real_output = discriminator(hr.numpy())

                        d_loss = d_loss_fn(discriminator, g_output, hr, flag_only_D)
                        #d_loss, _ = discriminator.train_on_batch(g_output, d_real_output, reset_metrics=False)
                        d_loss_history.append(d_loss)

                        with d_tape.stop_recording():
                            d_gradients = d_tape.gradient(d_loss, discriminator.variables)
                            d_optimizer.apply_gradients(zip(d_gradients, discriminator.variables))
                            print('Discriminator Loss:{}'.format(d_loss))

                if step_counter % 200 == 0 and flag_only_D == False:

                    save_path_d = ckpt_manager_d.save()
                    save_path_g = ckpt_manager_g.save()
                    print("\n----------Saved Discriminator for step {}: {}-------------\n"
                          "\n----------Saved Generator for step {}: {}-------------\n"
                          .format(step_counter, save_path_d, step_counter, save_path_g))

                    f = open('g_loss.txt', 'a')
                    for g in g_loss_history:
                        f.write(str(g))
                        f.write('\n')
                    f.close()
                    g_loss_history = []

                    f = open('d_loss.txt', 'a')
                    for d in d_loss_history:
                        f.write(str(d))
                        f.write('\n')
                    f.close()
                    d_loss_history = []

                if step_counter % 301 == 0 and flag_only_D == False:
                    # export evaluating parameters for [7,:,:] in a current patch
                    score_patch(generator(lr.numpy()), hr, 7, CUBE=CUBE)

                if step_counter % 701 == 0 and flag_only_D == False:
                    display.clear_output(wait=True)

                if step_counter == 55000:
                    print('\n ----------------- Completed for 55k steps! --------------------------\n')
                    break
            if step_counter == 55000:
                break
        if step_counter == 55000:
            break
