# Medical Super Resolution Learning：mDCSRN

This is an implement of mDCSRN with Tensorflow 2.0: [Efficient and Accurate MRI Super-Resolution Using a Generative Adversarial Network and 3D Multi-level Densely Connected Network](https://arxiv.org/ftp/arxiv/papers/1803/1803.01417.pdf)

##  - For Training
**arg_parser.py: **set training parameters, see details in the code.
**mdcsrn.py: **training code, including: pretrain(solely training for Generator), WGAN-Network(training for both Generator and Discriminator at the certain steps)

## - Cur-Version Data Formation
in the directory **data**, a **3d_hr_data.npy** and its corresponding **3d_lr_data.npy** should be prepared

