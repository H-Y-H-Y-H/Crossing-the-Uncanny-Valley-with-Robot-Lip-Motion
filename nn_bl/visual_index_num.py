import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

d_root = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/'


demo_id = 2
nn_index_id = np.loadtxt(f'nn_lmks_id_{demo_id}.csv')
save_root = f'/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/data1001/nn_bl/{demo_id}/'
os.makedirs(save_root,exist_ok=True)
img_path = d_root + f'data1001/vae/img/'


for i in range(len(nn_index_id)):
    idx = nn_index_id[i]
    img_read_synthesized = plt.imread(img_path + '/%d.png' % idx)  # [:480,:640]

    plt.imsave(save_root+'%d.png' % i, img_read_synthesized)

