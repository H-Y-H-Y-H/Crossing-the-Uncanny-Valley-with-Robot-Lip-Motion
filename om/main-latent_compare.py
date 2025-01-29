import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import os
from tqdm import trange, tqdm
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

checkpoint = torch.load("Models/mix_modelV5-3_128.pt", map_location="cpu")
image_size = 128
batch_size = 128
latent_root = 'Latents/'
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                # transforms.RandomHorizontalFlip(0.5),
                                # transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])
syth_Encoder = VAE(channel_in=3,
              ch=2,
              blocks=(1,4,8,16,32),
              latent_channels=1,
              num_res_blocks=1,
              norm_type='bn',
              deep_model=False).to(device)

syth_Encoder.load_state_dict(checkpoint['model_state_dict'])
syth_Encoder.eval()


demo_id = 2

# "0" collect latents_synthesized
# '1' collect latents_om
# '2' collect latents_lmks_min_dist
mode = 3

if mode == 0: # Synthesized
    test_dataset_root = f'../../EMO_GPTDEMO/robot_data/data1001/vae/{demo_id}/'
    latent_save_root = latent_root+'syn/'
elif mode == 1: # om
    test_dataset_root = f'Results/get_latent_image/{demo_id}/'
    latent_save_root = latent_root+'om/'
elif mode == 2: # audio_bl
    test_dataset_root =  f'../../EMO_GPTDEMO/robot_data/output_cmds/wav_bl/{demo_id}/'
    latent_save_root = latent_root+'wav_bl/'

else: # nn_lmks
    test_dataset_root = f'../../EMO_GPTDEMO/robot_data/data1001/nn_bl/{demo_id}/'
    latent_save_root = latent_root+'nn_baseline/'

# all_dataset_lantent real images
latent_dataset = np.load(latent_root+'latent-V5-3.npy')

test_dataset = Recons_data(test_dataset_root=test_dataset_root, transform=transform)
data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


latent_list = [] # latent space
latent_nn_bl_list = []
count = 0
with torch.no_grad():
    with torch.cuda.amp.autocast():
        for i, bundle in enumerate(tqdm(data_loader, leave=False)):
            test_images = bundle['img']
            recon_img, mu, log_var = syth_Encoder(test_images.to(device))
            # recon_img2 = vae_net.run_decoder(mu)

            mu = mu.detach().cpu().numpy()
            latent = mu.reshape(mu.shape[0], -1)
            latent_list.append(latent)

np.save(latent_save_root+f'latent_{demo_id}.npy',np.concatenate(latent_list))



