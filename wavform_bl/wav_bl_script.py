# visualize_dataset()
import os

import cv2

xy1 = (310, 250 - 30)
xy2 = (438, 378 - 30)

d_root = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/'

def video_2_frames(video_source, crop_flag=False,save_fig = False,save_path = './'):
    cap = cv2.VideoCapture(video_source)
    list_img = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if crop_flag:
            frame = frame[:, 80:560]
        cv2.imshow('frame', frame)
        if save_fig:
            if count >25:
                frame = frame[xy1[1]:xy2[1], xy1[0]:xy2[0]]
                cv2.imwrite(save_path + "%d.png"%(count-26), frame)
            count +=1
        list_img.append(frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return list_img

demo_id = 2
video_source = d_root+ f'output_cmds/wav_bl/{demo_id}_wavbl.mp4'
save_root = d_root+ f'output_cmds/wav_bl/{demo_id}/'
os.makedirs(save_root,exist_ok=True)

video_2_frames(video_source,save_fig=True,save_path=save_root)


