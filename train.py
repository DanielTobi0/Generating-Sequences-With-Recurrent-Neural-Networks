from utils import load_data, inputTensor

train_data_path = 'data/ptbdataset/ptb.train.txt'
test_data_path = 'data/ptbdataset/ptb.test.txt'
validation_data_path = 'data/ptbdataset/ptb.valid.txt'

train_data = load_data(train_data_path)
test_data = load_data(test_data_path)
validation_data = load_data(validation_data_path)

all_letters = list(set(train_data))
all_letters = ''.join(all_letters)
n_letters = len(all_letters)

tensor_train_data = inputTensor(train_data[:5000], n_letters, all_letters)
tensor_test_data = inputTensor(test_data[:5000], n_letters, all_letters)
tensor_validation_data = inputTensor(validation_data[:5000], n_letters, all_letters)
