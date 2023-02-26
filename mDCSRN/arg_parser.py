import argparse

def training_parser():
    parser = argparse.ArgumentParser(description='Training arguments.')
    
    parser.add_argument('-glr', '--generator_learning_rate', action='store',
                         default=1e-4, type=float,
                         help=('Generator Learning Rate. Default: '
                               '0.0001'))
    parser.add_argument('-bs', '--batch_size', action='store', 
                         default=2, type=int,
                         help='Batch Size. Default: "2"')
    parser.add_argument('-ps', '--patch_size', action='store',
                        default=2, type=int,
                        help='Patch Size. Default: "2"')
    parser.add_argument('-ep', '--epochs', action='store', default=50,
                         type=int, help=('Epochs. Default: 50'))
    parser.add_argument('-csv_path', '--data_csv_path', action='store',
                        default='./info.csv', type=str,
                         help=('Data CSV Path. Default: "./info.csv"'))

    return parser
