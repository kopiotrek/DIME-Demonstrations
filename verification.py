import os
import cv2
import pickle

def depickle_video(demo_folder_path):
    states = os.listdir(demo_folder_path)

    video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 5, (1280, 720))
    #  = cv2.VideoWriter('demo_video.avi', -1, 1, (1280, 720))

    states.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    iter = 0

    for state in states:
        state_path = os.path.join(demo_folder_path, state)

        file = open(state_path, 'rb')

        state_content = pickle.load(file)

        
        image = state_content['image']
        
        if iter == 0:
            print(state_content)

        # print('Obtained image from: {}'.format(state))
        iter += 1
        cv2.imshow('Recording', image)
        cv2.waitKey(1)

        video.write(image)


    video.release()

    cv2.destroyAllWindows()