from arg_parser import training_parser
from training import training_loop
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
	args = training_parser().parse_args()

	LR_G             = args.generator_learning_rate
	LR_D			 = args.discriminator_learning_rate
	EPOCHS           = args.epochs
	LOSS_FUNC        = args.loss_type
	BATCH_SIZE       = args.batch_size
	EPOCH_START      = args.epoch_start
	N_TRAINING_DATA  = args.n_training_data
	training_loop(LR_G, LR_D, EPOCHS, BATCH_SIZE, N_TRAINING_DATA, LOSS_FUNC, EPOCH_START)


if __name__ == '__main__':
    main()
