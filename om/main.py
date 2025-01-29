import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse
import cv2
from RES_VAE_Dynamic import VAE




class Recons_data(Dataset):
    def __init__(self, test_dataset_root,transform):
        self.root = test_dataset_root
        self.transform = transform

    def __getitem__(self, idx):
        image = plt.imread(self.root + '%d.png'%idx)[...,:3]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)

        sample = {"img": image.to(device, dtype=torch.float)}
        return sample

    def __len__(self):
        num_files = os.listdir(self.root)
        return len(num_files)


mode = 1
# "0" collect latents through encoder
# '1' generate test based on min dist latent
model_version="V5-2"
checkpoint = torch.load(f"Models/mix_model{model_version}_128.pt", map_location="cpu")
# checkpoint = torch.load("Models/finetune_modelV4_128.pt", map_location="cpu")

demo_id = 20
image_size = 128
batch_size = 128
latent_root = 'Latents/'


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                # transforms.RandomHorizontalFlip(0.5),
                                # transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

test_dataset_root = f'../../EMO_GPTDEMO/robot_data/data1001/vae/{demo_id}/'
latent_dist_root = latent_root+'latent_dist/'

dataset_root = f'../../EMO_GPTDEMO/robot_data/data1001/vae/lip/'


os.makedirs(latent_root, exist_ok=True)

test_dataset = Recons_data(test_dataset_root=test_dataset_root, transform=transform)
all_dataset = Recons_data(test_dataset_root=dataset_root, transform=transform)

if mode == 0:
    data_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
elif mode == 1 or mode ==2:
    data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    latent_dataset = np.load(latent_root+f'latent-{model_version}.npy')


syth_Encoder = VAE(channel_in=3,
              ch=2,
              blocks=(1,4,8,16,32),
              latent_channels=1,
              num_res_blocks=1,
              norm_type='bn').to(device)

syth_Encoder.load_state_dict(checkpoint['model_state_dict'])
syth_Encoder.eval()

latent_synth_list = [] # latent space of synthesized images
latent_space_list = [] # latent space of all dataset images
indices_list = []
dist_list = []
latent_list = []
count = 0
with torch.no_grad():
    with torch.cuda.amp.autocast():
        for i, bundle in enumerate(tqdm(data_loader, leave=False)):
            test_images = bundle['img']
            recon_img, mu, log_var = syth_Encoder(test_images.to(device))
            # recon_img2 = vae_net.run_decoder(mu)

            mu = mu.detach().cpu().numpy()
            latent = mu.reshape(mu.shape[0], -1)
            if mode == 1:
                image_list = []
                for j in range(len(latent)):
                    print(j)
                    latent_synth_list.append(latent[j])
                    dist = np.sum((latent[j]-latent_dataset)**2,axis=1)
                    idx = np.argmin(dist)

                    dist_list.append(dist[idx])
                    latent_list.append(latent_dataset[idx])

                    indices_list.append(idx)
                    sele_img = cv2.imread(dataset_root+f'/%d.png'%idx)[...,:3]
                    cv2.putText(sele_img, '%s'%count, (20,110), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                                color=[255,255,255], thickness=2)

                    cv2.putText(sele_img, '%s'%idx, (65,110), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=[255,255,255], thickness=2)

                    sele_img = cv2.cvtColor(sele_img, cv2.COLOR_RGB2BGR)
                    count += 1
                    sele_img = sele_img/255
                    # # Display the image
                    # cv2.imshow('Image with Text', sele_img)
                    # cv2.waitKey(0)  # Wait for a key press to close the displayed image
                    # cv2.destroyAllWindows()

                    sele_img = sele_img.transpose((2, 0, 1))
                    sele_img = torch.from_numpy(sele_img).to(device, dtype=torch.float)
                    sele_img = transform(sele_img)

                    image_list.append(sele_img)
                image_list = torch.stack(image_list, dim=0)

                img_cat = torch.cat((image_list, test_images), 2).float()
                vutils.save_image(img_cat, "%s/nn_latent%d.png" % ("Results", i),normalize=True)

            elif mode==0:
                latent_space_list.append(latent)

            else:
                img_cat = torch.cat((recon_img, test_images), 2).float()
                vutils.save_image(img_cat,
                                  "%s/mix_model%d.png" % ("Results", i),
                                  normalize=True)

        indices_list = np.asarray([indices_list]).T
        indices_list = np.hstack((np.arange(0, len(indices_list)).reshape(len(indices_list), 1), indices_list))
        np.savetxt(f'Results/index_num{demo_id}.csv', indices_list, fmt='%i')

        np.savetxt(f'Latents/latent_dist/dist_om_{demo_id}.csv', dist_list)
        np.save(f'Latents/om/latent_{demo_id}.npy', latent_list)


if mode == 0:
    np.save(latent_root+f'latent-{model_version}.npy',np.concatenate(latent_space_list))
elif mode == 1:
    np.save(f'Latents/syn/latent_{demo_id}.npy', latent_synth_list)

# img = plt.imread(data_path+'395.png')
# xy1 = (310,250)
# xy2 = (438,378)
#
# plt.imshow(img[xy1[1]:xy2[1], xy1[0]:xy2[0]])
# plt.show()
#

