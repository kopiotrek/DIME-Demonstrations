import os
import pickle
import cv2

def check_dir(dir):
    if not os.path.exists(dir):
        print('Making directory: {}'.format(dir))
        os.makedirs(dir)
    else:
        print('{} - Directory already exists'.format(dir))

class ImageExtractor():
    def __init__(self, data_path, storage_path):
        self.data_path = data_path
        self.storage_path = storage_path

        check_dir(self.storage_path)

    def getting_data_dir_list(self, dir):
        files = os.listdir(dir)
        files.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        return files

    def get_image_data(self, demo_dir, demo_idx):
        # Getting the list of data        
        state_list = self.getting_data_dir_list(demo_dir)
        
        demo_images_folder_path = os.path.join(self.storage_path, "demo_images_{}".format(demo_idx))
        check_dir(demo_images_folder_path)

        for idx, state in enumerate(state_list):
            state_path = os.path.join(demo_dir, state)
            state_file = open(state_path, 'rb')
            state_data = pickle.load(state_file)

            extracted_demo_image_path = os.path.join(demo_images_folder_path, "state_{}.jpg".format(idx))
            cv2.imwrite(extracted_demo_image_path, state_data['robot_image'])

    def get_all_demo_image_data(self):
        demos_list = os.listdir(self.data_path)

        for idx, demo in enumerate(demos_list):
            demo_path = os.path.join(self.data_path, demo)

            self.get_image_data(demo_path, idx + 1)

            demo_images_folder_path = os.path.join(self.storage_path, "demo_images_{}".format(idx + 1))
            print("Number of images extracted from the demo: {}".format(len(os.listdir(demo_images_folder_path))))

if __name__ == "__main__":
    extractor = ImageExtractor("/home/sridhar/dexterous_arm/demonstrations/original_data/fidget_spinning", "/home/sridhar/dexterous_arm/demonstrations/image_data/fidget_spinning")
    extractor.get_all_demo_image_data()