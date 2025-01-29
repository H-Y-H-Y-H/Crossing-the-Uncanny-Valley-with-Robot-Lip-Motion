import numpy as np
import wandb
import argparse
import matplotlib.pyplot as plt
import torch
import os
import cv2

# Check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("start", device)

init_cmds = np.asarray([0.5, 0.625, 0.53333, 0.1, 1.00000])

lips_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 78, 191, 80,
            81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
inner_lips_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

select_lmks_id = lips_idx + inner_lips_idx
key_cmds = np.asarray([0,1,2,4,6])

d_root = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/'
data_path = d_root + 'data1001/'

dataset_lmks = np.load(data_path+'m_lmks.npy')
dataset_cmd = np.loadtxt(data_path+'action.csv')

dataset_cmd = dataset_cmd[:, key_cmds]

mouth_dataset_lmks = dataset_lmks[:,select_lmks_id]

# zero index is shifted too much.
init_lmks = np.asarray(mouth_dataset_lmks[1])

lmks_max = mouth_dataset_lmks.max(0)
lmks_min = mouth_dataset_lmks.min(0)


if __name__ == '__main__':
    # for i in range(6):
    #     init_lmks = np.asarray(mouth_dataset_lmks[i])
    #     plt.scatter(init_lmks[:, 0], init_lmks[:, 1])
    # plt.show()
    import shutil

    syn_data_root = '/Users/yuhan/PycharmProjects/EMO_GPTDEMO/robot_data/data1001/vae/'
    num_data = 0
    for i in range(22):
        num_data += len(os.listdir(syn_data_root+'%d'%i))
    print(num_data)
    quit()
    # data_path = d_root + 'data1001/'
    # save_path = data_path+'vae/lip/'
    idx = 20
    sync_data_path = d_root + 'data1001/synthesized/lmks_rendering/%d/'%idx

    data_path = d_root + 'output_cmds/1028_one_c_one_pic/img%d/'%idx


    save_path = d_root + 'output_cmds/1028_one_c_one_pic/img_lip%d/'%idx
    img_num = len(os.listdir(data_path))

    os.makedirs(save_path,exist_ok=True)
    xy1 = (310,250-30)
    xy2 = (438,378-30)

    # for i in range(img_num):
    #     print(i)
    #
    #     img_sync = plt.imread(sync_data_path+'%d.png'%i)[:480,:480]
    #     img = plt.imread(data_path + '%d.png' % i)
    #     img_rect = img[:,(640-480)//2:(640+480)//2]
    #
    #     img_lip = img[xy1[1]:xy2[1], xy1[0]:xy2[0]]
    #     img_lip = cv2.resize(img_lip,dsize=(480,480))
    #
    #
    #
    #     img = np.concatenate((img_sync,img_rect,img_lip),axis=0)
    #
    #     plt.imsave(save_path+'%d.png'%i,img)


