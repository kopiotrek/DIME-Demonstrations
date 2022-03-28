import os
import argparse
from xxlimited import new

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str)

if __name__ == "__main__":
    options = parser.parse_args()

    new_data_path = os.path.join(os.getcwd(), "fresh_data", options.task, "success")

    demos = os.listdir(new_data_path)
    destination_len = len(os.listdir(os.path.join(os.getcwd(), "original_data", options.task)))

    for idx, demo in enumerate(demos):
        demo_path = os.path.join(new_data_path, demo)
        new_demo_path = os.path.join(new_data_path, "demonstration_{}".format(idx + destination_len + 1))

        os.rename(demo_path, new_demo_path)