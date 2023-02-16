import argparse

def training_parser():
    parser = argparse.ArgumentParser(description='Training arguments.')
    
    parser.add_argument('-glr', '--generator_learning_rate', action='store',
                         default=1e-4, type=float,
                         help=('Generator Learning Rate. Default: '
                               '0.0001'))
    parser.add_argument('-dlr', '--discriminator_learning_rate', action='store',
                        default=1e-4, type=float,
                        help=('Discriminator Learning Rate. Default: '
                              '0.0001'))
    parser.add_argument('-bs', '--batch_size', action='store', 
                         default=1, type=int,
                         help='Batch Size. Default: "1"')
    parser.add_argument('-ep', '--epochs', action='store', default=50,
                         type=int, help=('Epochs. Default: 50'))
    parser.add_argument('-eps', '--epoch_start', action='store', default=0, 
                         type=int, help=('Starting Epoch. Default: 0'))
    parser.add_argument('-lt', '--loss_type', action='store', default='l1_loss',
                         type=str, choices=['l1_loss', 'l2_loss'],
                         help=('Loss type, either L1 (MAE) or L2 (MSE). Default: L1'))
    parser.add_argument('-ntd', '--n_training_data', action='store', default=1000,
                         type=int, help=('Number of training data used each epoch. Default: 1000'))

    return parser
