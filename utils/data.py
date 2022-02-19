import os
import pickle

def check_dir(dir):
    if not os.path.exists(dir):
        print('Making directory: {}'.format(dir))
        os.makedirs(dir)
    else:
        print('Directory already exists {}'.format(dir))

def store_pickle_data(pickle_path, data):
    file = open(pickle_path, 'ab')
    pickle.dump(data, file)
    file.close()