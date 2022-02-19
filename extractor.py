import os
import argparse
from core.demo_collector import DemoCollector

BASE_DIR = os.path.join(os.getcwd(), "fresh_data")

collect_data_parser = argparse.ArgumentParser()
collect_data_parser.add_argument('-t', '--task', type=str)
collect_data_parser.add_argument('-f', '--frequency', default=3, type=int)

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
    check_dir(os.path.join(storage_dir, "success"))
    check_dir(os.path.join(storage_dir, "failure"))

    success_demos, failure_demos = len(os.listdir(os.path.join(storage_dir, "success"))), len(os.listdir(os.path.join(storage_dir, "failure"))) 

    return storage_dir, success_demos + failure_demos + 1

if __name__ == '__main__':
    options = collect_data_parser.parse_args()

    storage_dir, demo_number = check_structure(options.task, BASE_DIR)

    print("Initializing extractor for task: {}".format(options.task))
    extractor = DemoCollector(options.task, storage_dir, options.frequency)

    print("Collecting demonstration number: {}".format(demo_number))
    extractor.collect_data(demo_number)
