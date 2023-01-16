import tensorflow as tf
from logger import log
from model import Generator, Discriminator

def get_generator(PATCH_SIZE, LR_G):
    
    generator_g         = Generator(PATCH_SIZE)
    generator_optimizer = tf.keras.optimizers.Adam(LR_G)

    path = "tensor_checkpoints_g/"

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                               generator_optimizer=generator_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        log('Generator: Latest checkpoint restored!!')
    else:
        log("Generator: No checkpoint found! Staring from scratch!")

    return generator_g, generator_optimizer, ckpt_manager

def save_generator(ckpt_manager, epoch):
    ckpt_save_path = ckpt_manager.save()
    log('Saving checkpoint_g for epoch {} at {}'.format(epoch, ckpt_save_path))


def get_discrimitor(PATCH_SIZE, FILTERS_NUM, LR_D):

    d = Discriminator(PATCH_SIZE, FILTERS_NUM)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_D)

    path = "tensor_checkpoints_d/"

    ckpt = tf.train.Checkpoint(discriminator=d,
                               d_optimizer=d_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        log('Discriminator: Latest checkpoint restored!!')
    else:
        log("Discriminator: No checkpoint found! Staring from scratch!")

    return d, d_optimizer, ckpt_manager

def save_discriminator(ckpt_manager, epoch):
    ckpt_save_path = ckpt_manager.save()
    log('Saving checkpoint_d for epoch {} at {}'.format(epoch, ckpt_save_path))