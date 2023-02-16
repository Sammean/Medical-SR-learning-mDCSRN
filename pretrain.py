import os
from datetime import time
import tensorflow as tf
from IPython import display
import numpy as np
from mDCSRN.dataset import load_idset, DatafromCSV
from mDCSRN.model import Generator
from mDCSRN.loss_functions import supervised_loss


train_set, test_set, val_set, eval_set = load_idset('./t1-mx3d.csv')
# ------------------ get one sample -------------------------------
sample_id = val_set.take(1)
sample_indices=np.array([idx.numpy() for a, idx in enumerate(sample_id)])
Sampleloader = DatafromCSV(sample_indices)
sample_set = Sampleloader.load_patchset()

# ========================== Initial Train =========================== #
# ----- The Loss is need further validation: from valid images --------#
#------------------------ Hyper parameters ------------------- #
batch_size = 2
patch_size = 2
n_epochs = 10
BUFFER_SIZE = 777
n_patches = np.ceil(480/64)*np.ceil(672/64)*np.ceil(552/64)*batch_size
learning_rate = 1e-4
beta1 = 0.5

#-------------------------- Optimizer ----------------------- #
g_optimizer_init = tf.train.AdamOptimizer(learning_rate,beta1=beta1)

generator = Generator()
generator.compile(loss=supervised_loss,
                  optimizer=g_optimizer_init ,
                  metrics =['mse'])


#-------------------------- Checkpoint ----------------------- #
# Works On Colab
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(g_optimizer_init=g_optimizer_init,
                                 generator=generator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#--------------- Marker for Loops Initialization ------------- #
batch_counter = 0
step_counter = 0
g_loss_history = []
valid_loss_history = []

for epoch in range(n_epochs):
    start = time.time()
    # ------------------- Split batch in idset ---------------------
    train_set = train_set.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size)
    iter = train_set.make_one_shot_iterator()
    for el in iter:
        batch_counter += 1
        display.clear_output(wait=True)
        indices=np.array(el)
    # ---------------------- Get data from id ----------------------
        Batchloader = DatafromCSV(indices)
        train_dataset = Batchloader.load_patchset()
    # -------------------- Split patch in dataset ------------------
        train_dataset = train_dataset.shuffle(buffer_size=n_patches).batch(patch_size)
        for patch, (lr, hr) in enumerate(train_dataset):
            step_counter += 1
            print('Global Step:{}, Subject No.{}'.format(step_counter,batch_counter))
            g_loss,_ = generator.train_on_batch(lr,hr,reset_metrics=False)
            g_loss_history.append(g_loss)

            if step_counter % 100 == 0:
                manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
                save_path = manager.save()
                print("\n----------Saved checkpoint for step {}: {}-------------\n".format(step_counter, save_path))
            if step_counter == 50000:
                print('\n ----------------- Completed for 50k steps! --------------------------\n')
                break
        if step_counter == 50000:
          break
    if step_counter == 50000:
      break