import os
import sys
import time
import signal
import datetime
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from logger import log
from model import Generator
from loss_functions import supervised_loss, d_loss_fn
from plotting import generate_images, plot_losses
from data_preparing import get_batch_data
from model_checkpoints import get_generator, save_generator, get_discrimitor, save_discriminator


def signal_handler(sig, frame):
    stop_log = "The training process was stopped at "+time.ctime()
    log(stop_log)
    plot_losses(epochs_plot, total_generator_g_error_plot)
    save_generator(ckpt_manager_g, "final_epoch")
    save_discriminator(ckpt_manager_d, "final_epoch")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


@tf.function
def train_step_d(real_x, real_y, gp_lam=10.0, gan_lam=1e-3):
    
    with tf.GradientTape(persistent=True) as tape:

        fake_y = generator_g(real_x, training=True) # SR

        loss_intensity = supervised_loss(real_y, fake_y)
        loss_gan = d_loss_fn(discriminator_d, fake_y, real_y, training=True, lam=gp_lam)
        loss = loss_intensity + gan_lam * loss_gan

    grads = tape.gradient(loss, discriminator_d.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(grads, discriminator_d.trainable_variables))

    return loss


@tf.function
def train_step_g(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        gen_g_super_loss = supervised_loss(real_y, fake_y)

    gradients_of_generator = tape.gradient(gen_g_super_loss, generator_g.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_g.trainable_variables))

    return gen_g_super_loss


def training_loop(LR_G, LR_D, EPOCHS, BATCH_SIZE, N_TRAINING_DATA, LOSS_FUNC, EPOCH_START):

    begin_log = '### Began training at {} with parameters: Starting Epoch={}, Epochs={}, Batch Size={}, Training Data={}, Learning Rate={}, Loss Function={}'.format(time.ctime(), EPOCH_START, EPOCHS, BATCH_SIZE, N_TRAINING_DATA, LR_G, LOSS_FUNC)
    log(begin_log)

    training_start = time.time()

    lr_data = np.load('data/3d_lr_data.npy') # (N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1)
    hr_data = np.load('data/3d_hr_data.npy') # (N, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1)
    # print(lr_data.shape, hr_data.shape)
    PATCH_SIZES = hr_data.shape[1:4]         # (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    assert(PATCH_SIZES==lr_data.shape[1:4])
    PATCH_SIZE  = PATCH_SIZES[0]
    assert(PATCH_SIZE==PATCH_SIZES[1]==PATCH_SIZES[2])

    patch_log = "Patch Size is "+str(PATCH_SIZE)
    log(patch_log)

    global generator_g, generator_optimizer, ckpt_manager_g, discriminator_d, discriminator_optimizer, ckpt_manager_d
    generator_g, generator_optimizer, ckpt_manager_g = get_generator(PATCH_SIZE, LR_G)
    discriminator_d, discriminator_optimizer, ckpt_manager_d= get_discrimitor(PATCH_SIZE, 64, LR_D)

    comparison_image    = 725
    comparison_image_hr = hr_data[comparison_image]
    comparison_image_lr = lr_data[comparison_image]

    generate_images(generator_g, comparison_image_lr, comparison_image_hr, PATCH_SIZE, "a_first_plot")

    global epochs_plot, total_generator_g_error_plot
    epochs_plot = []
    total_generator_g_error_plot = []
    total_discriminator_d_error_plot = []

    for epoch in range(EPOCH_START, EPOCH_START+EPOCHS):

        epoch_s_log = "Began epoch "+str(epoch)+" at "+time.ctime()
        log(epoch_s_log)

        epoch_start = time.time()
        
        data_x = lr_data[0:N_TRAINING_DATA]
        data_y = hr_data[0:N_TRAINING_DATA]
        
        for i in range(0, N_TRAINING_DATA, BATCH_SIZE):
            r = np.random.randint(0,2,3)
            batch_data = get_batch_data(data_x, i, BATCH_SIZE, r[0], r[1], r[2])
            batch_label = get_batch_data(data_y, i, BATCH_SIZE, r[0], r[1], r[2])
            #generator_loss = train_step_g(batch_data, batch_label).numpy()
            discriminator_loss = train_step_d(batch_data, batch_label).numpy()

        epochs_plot.append(epoch)
        #total_generator_g_error_plot.append(generator_loss)
        total_discriminator_d_error_plot.append(discriminator_loss)
                
        comparison_image_hr = hr_data[comparison_image]
        comparison_image_lr = lr_data[comparison_image]

        generate_images(generator_g, comparison_image_lr, comparison_image_hr, PATCH_SIZE, "epoch_"+str(epoch) ," Epoch: "+str(epoch) )
        
        # epoch_e_log = "Finished epoch "+str(epoch)+" at "+time.ctime()\
        #               +". G Loss = "+str(generator_loss)+"."+". D Loss = "+str(discriminator_loss)
        epoch_e_log = "Finished epoch " + str(epoch) + " at " + time.ctime() + ". D Loss = " + str(discriminator_loss)
        log(epoch_e_log)

        epoch_seconds = time.time() - epoch_start
        epoch_t_log = "Epoch took "+str(datetime.timedelta(seconds=epoch_seconds))
        log(epoch_t_log)

        hr_data, lr_data = shuffle(hr_data, lr_data)
        if (epoch + 1) % 20 == 0:
            #save_generator(ckpt_manager_g, epoch)
            save_discriminator(ckpt_manager_d, epoch)


    plot_losses(epochs_plot, total_generator_g_error_plot)
    generate_images(generator_g, comparison_image_lr, comparison_image_hr, PATCH_SIZE, "z_final_plot")
    
    training_e_log = "Finished training at "+time.ctime()
    log(training_e_log)

    training_seconds = time.time() - training_start
    training_t_log = "Training took "+str(datetime.timedelta(seconds=training_seconds))
    log(training_t_log)
