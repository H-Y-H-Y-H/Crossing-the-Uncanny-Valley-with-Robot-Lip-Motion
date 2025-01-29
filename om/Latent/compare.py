import os
import numpy as np
import matplotlib.pyplot as plt
import random


random.seed(0)
demo_id = 1

l_syn = np.load(f'syn/latent_{demo_id}.npy')
l_om = np.load(f'om/latent_{demo_id}.npy')
l_bl = np.load(f'nn_baseline/latent_{demo_id}.npy')
l_wav_bl = np.load(f'wav_bl/latent_{demo_id}.npy')

all_latent = np.load('latent-V5-3.npy')
randoms_list = random.choices(range(20000), k=len(l_syn))
l_rand = all_latent[randoms_list]
np.save(f'random/latent_{demo_id}.npy',l_rand)

l_om_shift_move_15f = np.concatenate((np.asarray([l_om[0]]*15),l_om[:-15]))
l_om_shift_move_1f = np.concatenate((np.asarray([l_om[0]]*1),l_om[:-1]))


dist_om = np.mean((l_om - l_syn) ** 2, axis=1)  # (300,)
dist_bl = np.mean((l_bl - l_syn) ** 2, axis=1)  # (300,)
dist_shift = np.mean((l_om_shift_move_15f - l_syn) ** 2, axis=1)  # (300,)
dist_shift1f = np.mean((l_om_shift_move_1f - l_syn) ** 2, axis=1)  # (300,)

dist_shuffle = np.mean((l_rand - l_syn) ** 2, axis=1)  # (300,)

dist_wav_bl = np.mean((l_wav_bl - l_syn) ** 2, axis=1)  # (300,)

# OM, NN_Lmks, Shift1, shift15, random, wav_bl
dist_list = [dist_om,dist_bl, dist_wav_bl,dist_shift1f, dist_shift, dist_shuffle]
data_save = []
for i in range(len(dist_list)):
    dist_data = dist_list[i]
    data_mean = np.mean(dist_data)
    data_std = np.std(dist_data)
    data_min = np.min(dist_data)
    data_max = np.max(dist_data)
    da = [data_mean,data_std,data_min,data_max]
    data_save.append(da)
np.savetxt(f'paper_data_{demo_id}.csv',np.asarray(data_save).T)

plt.plot(dist_om,c='blue')
plt.plot(dist_bl,c='red')
plt.plot(dist_shift,c='green')
plt.plot(dist_shift1f,c='black')
plt.plot(dist_wav_bl,c='pink')
plt.plot(dist_shuffle,c='orange')
plt.show()

