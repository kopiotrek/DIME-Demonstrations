import os
import cv2
import pickle

class VideoGenerator():
    def __init__(self, data_path, storage_path):
        self.data_path = data_path
        self.demo_list = os.listdir(self.data_path)
        self.demo_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.storage_path = storage_path

        if not os.path.exists(self.storage_path):
            print('Making directory: {}'.format(self.storage_path))
            os.makedirs(self.storage_path)
        else:
            print('Directory already exists')

    def generate_video_for_demo(self, demo_path, demo_idx):
        states_list = os.listdir(demo_path)
        states_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        demo_video_path = os.path.join(self.storage_path, "demo_{}.avi".format(demo_idx))
        demo_video = cv2.VideoWriter(demo_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 5, (1280, 720))

        for state in states_list:
            state_path = os.path.join(demo_path, state)
            state_file = open(state_path, 'rb')
            state_data = pickle.load(state_file)
            image = state_data['robot_image']
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            demo_video.write(image)

        demo_video.release()

    def generate_videos(self):
        for idx, demo in enumerate(self.demo_list):
            print("Generating video for {}".format(demo))
            demo_path = os.path.join(self.data_path, demo)
            self.generate_video_for_demo(demo_path, idx + 1)

if __name__ == "__main__":
    v = VideoGenerator(data_path = "/home/sridhar/dexterous_arm/demonstrations/resampled_data_sample/fidget_spinning/2_cm_data", storage_path = "/home/sridhar/dexterous_arm/demonstrations/videos/")
    v.generate_videos()