import os
import argparse
from typing import final
from core.demo_resampler import StateExtractor
from core.feature_extractor import FeatureExtractor

RESAMPLE_BASE_DIR = os.path.join(os.getcwd(), "resampled_data_sample")
FINAL_BASE_DIR = os.path.join(os.getcwd(), "final_data_sample")
# INPUT_BASE_DIR = os.path.join(os.getcwd(), "original_data")
INPUT_BASE_DIR = os.path.join(os.getcwd(), "temp_data")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str)
parser.add_argument('-d', '--delta', default=2, type=int)

def check_dir(storage_path):
    if not os.path.exists(storage_path):
        print('Making directory: {}'.format(storage_path))
        os.makedirs(storage_path)
    else:
        print('{} - Directory already exists'.format(storage_path))

def check_structure(task, base_dir):
    check_dir(base_dir)

    if task == "rotate":
        folder_name = "cube_rotation"
    elif task == "flip":
        folder_name = "object_flipping"
    elif task == "spin":
        folder_name = "fidget_spinning"
    
    storage_dir = os.path.join(base_dir, folder_name)
    check_dir(storage_dir)

    return storage_dir

if __name__ == '__main__':
    options = parser.parse_args()

    input_dir = check_structure(options.task, INPUT_BASE_DIR)
    resample_dir = check_structure(options.task, RESAMPLE_BASE_DIR)
    resample_delta_dir = os.path.join(resample_dir, "{}_cm_data".format(options.delta))
    final_dir = check_structure(options.task, FINAL_BASE_DIR)
    final_delta_dir = os.path.join(final_dir, "final_data_{}_cm".format(options.delta))

    print("Initializing resampler for task: {}".format(options.task))
    resampler = StateExtractor(options.task, input_dir, resample_delta_dir, 0.01 * options.delta)
    print("Resampling demonstrations!")
    resampler.resample_demos()
    print("Resampling done! Data can be found in {}".format(resample_delta_dir))

    print("Initializing feature extractor for task: {}".format(options.task))
    feature_sampler = FeatureExtractor(options.task, resample_delta_dir, final_delta_dir)
    print("Extracting state-action pairs with images!")
    feature_sampler.get_all_demo_data()
    print("Extraction done! Data can be found in {}".format(final_delta_dir))