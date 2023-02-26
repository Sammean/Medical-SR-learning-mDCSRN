from arg_parser import training_parser
from train import pretrain_loop, train_loop
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
	args = training_parser().parse_args()

	LR               = args.generator_learning_rate
	EPOCHS           = args.epochs
	BATCH_SIZE       = args.batch_size
	PATCH_SIZE		 = args.patch_size
	CSV_PATH		 = args.data_csv_path
	pretrain_loop(CSV_PATH, BATCH_SIZE, PATCH_SIZE, EPOCHS, LR)

def main2():
	args = training_parser().parse_args()

	LR               = args.generator_learning_rate
	EPOCHS           = args.epochs
	BATCH_SIZE       = args.batch_size
	PATCH_SIZE		 = args.patch_size
	CSV_PATH		 = args.data_csv_path
	train_loop(CSV_PATH, BATCH_SIZE, PATCH_SIZE, EPOCHS)


if __name__ == '__main__':
    main()
