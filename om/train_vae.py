import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms as T

from vgg19 import VGG19
from RES_VAE_Dynamic import VAE
import numpy as np

model_name = 'mix_modelV5-3'
dataset_root = '../../EMO_GPTDEMO/robot_data/data1001/vae_finetune4/'
FINETUNE = True
Flag_pre_load_model = True
pretrain_file_name =  "mix_modelV4_128"
lr = 1e-6 * 5

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, default=model_name)
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, default=dataset_root)

parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default=".")
parser.add_argument("--norm_type", "-nt",
                    help="Type of normalisation layer used, BatchNorm (bn) or GroupNorm (gn)", type=str, default="bn")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=128)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=128)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=2)

parser.add_argument("--num_res_blocks", '-nrb',
                    help="Number of simple res blocks at the bottle-neck of the model", type=int, default=1)

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=1)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)
parser.add_argument("--block_widths", '-bw', help="Channel multiplier for the input of each block",
                    type=int, nargs='+', default=(1, 4, 8, 16, 32))
# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=lr)
parser.add_argument("--feature_scale", "-fs", help="Feature loss scale", type=float, default=1)
parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint", default=Flag_pre_load_model)
parser.add_argument("--deep_model", '-dm', action='store_true',
                    help="Deep Model adds an additional res-identity block to each down/up sampling stage")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(args.device_index if use_cuda else "cpu")
print("")


# Custom dataset class that returns two transformed images
class DoubleImageDataset(Dataset):
    def __init__(self, dataset, transform=None,finetune=False):
        self.dataset = dataset
        self.transform = transform
        self.finetune = finetune
        self.gt_folder = dataset_root


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get a single image and label from the dataset
        image1, label = self.dataset[idx]

        label_name = get_label_name(label)
        # If label == 1, load the second image from the '/0gt/' folder
        if ('gt' not in label_name) and self.finetune:
            # Generate the file path for the second image
            original_idx = self.dataset.indices[idx]
            img_name = os.path.split(self.dataset.dataset.samples[original_idx][0])[-1]  # Get the image name

            gt_folder = self.gt_folder + '/%sgt/'%label_name
            gt_img_path = os.path.join(gt_folder, img_name)

            # Load the second image from the folder
            image2 = Image.open(gt_img_path).convert("RGB")

            # Apply the transform to the loaded image (if transform is provided)
            if self.transform:
                image2 = self.transform(image2)

        else:
            image2 = image1

        # Return both images and the label
        return (image1, image2), label

# Create dataloaders
# This code assumes there is no pre-defined test/train split and will create one for you
print("-Target Image Size %d" % args.image_size)
transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                # transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

data_set = Datasets.ImageFolder(root=args.dataset_root, transform=transform)
class_to_idx = data_set.class_to_idx
# Inverse the mapping to get index-to-class mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}
# Example: Get the label name for the current label
def get_label_name(label_index):
    return idx_to_class.get(label_index, "Unknown label")

def func_kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


# Randomly split the dataset with a fixed random seed for reproducibility
test_split = 0.95
n_train_examples = int(len(data_set) * test_split)
n_test_examples = len(data_set) - n_train_examples
train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                    generator=torch.Generator().manual_seed(42))

# Wrap the dataset with the custom DoubleImageDataset to get two images per sample
train_set = DoubleImageDataset(train_set, transform=transform,finetune=FINETUNE)
test_set = DoubleImageDataset(test_set, transform=transform,finetune=FINETUNE)


train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True,num_workers=0)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(test_loader)
(test_images, gt_images), _ = next(dataiter)


def imshow(img_tensor, ax, title=None, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    img = img_tensor.clone().detach()  # Clone the tensor to avoid modifying the original one
    img = img.permute(1, 2, 0).numpy()  # Convert from [C, H, W] to [H, W, C]

    # If the image was normalized, denormalize it
    img = img * std + mean  # Reverse normalization assuming the image was normalized to [-1, 1]

    # Clip the image to ensure it's in the valid range [0, 1] for float
    img = img.clip(0, 1)

    ax.imshow(img)
    if title:
        ax.set_title(title)
    ax.axis('off')


# # Visualize the test images and ground truth images
# batch_size = 2
# fig, axs = plt.subplots(2, 20, figsize=(15, 6))
#
# for i in range(20):
#     # Plot test images (row 1)
#     imshow(test_images[i], axs[0, i])
#
#     # Plot ground truth images (row 2)
#     imshow(gt_images[i], axs[1, i])
#
# plt.tight_layout()
# plt.show()

# Create AE network.
vae_net = VAE(channel_in=test_images.shape[1],
              ch=args.ch_multi,
              blocks=args.block_widths,
              latent_channels=args.latent_channels,
              num_res_blocks=args.num_res_blocks,
              norm_type=args.norm_type).to(device)

# Setup optimizer
optimizer = optim.Adam(vae_net.parameters(), lr=args.lr)

# AMP Scaler
scaler = torch.cuda.amp.GradScaler()

if args.norm_type == "bn":
    print("-Using BatchNorm")
elif args.norm_type == "gn":
    print("-Using GroupNorm")
else:
    ValueError("norm_type must be bn or gn")

# Create the feature loss module if required
if args.feature_scale > 0:
    feature_extractor = VGG19().to(device)
    print("-VGG19 Feature Loss ON")
else:
    feature_extractor = None
    print("-VGG19 Feature Loss OFF")

# Let's see how many Parameters our Model has!
num_model_params = 0
for param in vae_net.parameters():
    num_model_params += param.flatten().shape[0]

print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))
fm_size = args.image_size//(2 ** len(args.block_widths))
print("-The Latent Space Size Is %dx%dx%d!" % (args.latent_channels, fm_size, fm_size))

os.makedirs(args.save_dir + "/Models",exist_ok=True)
os.makedirs(args.save_dir + "/Results",exist_ok=True)

# Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
# If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
# it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
save_file_name = args.model_name + "_" + str(args.image_size)

if args.load_checkpoint:
    if os.path.isfile(args.save_dir + "/Models/" + pretrain_file_name + ".pt"):
        checkpoint = torch.load(args.save_dir + "/Models/" + pretrain_file_name + ".pt",
                                map_location="cpu")
        print("-Checkpoint loaded!")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        vae_net.load_state_dict(checkpoint['model_state_dict'])

        if not optimizer.param_groups[0]["lr"] == args.lr:
            print("Updating lr!")
            optimizer.param_groups[0]["lr"] = args.lr

        start_epoch = checkpoint["epoch"]
        data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
    else:
        raise ValueError("Warning Checkpoint does NOT exist -> check model name or save directory")
else:
        print("Starting from scratch")
        start_epoch = 0
        # Loss and metrics logger
        data_logger = defaultdict(lambda: [])
print("")

min_loss= np.inf
patience = 0
threshold_patience=10

# Start training loop
for epoch in range(args.nepoch):
    print('Epoch: ', epoch)
    vae_net.train()
    for i, ((images, label_images), _) in enumerate(train_loader):
        images = images.to(device)
        label_images = label_images.to(device)

        bs, c, h, w = images.shape

        # We will train with mixed precision!
        with torch.cuda.amp.autocast():
            recon_img, mu, log_var = vae_net(images)

            kl_loss = func_kl_loss(mu, log_var)
            mse_loss = F.mse_loss(recon_img, label_images)
            loss = args.kl_scale * kl_loss + mse_loss

            # Perception loss
            if feature_extractor is not None:
                feat_in = torch.cat((recon_img, label_images), 0)
                feature_loss = feature_extractor(feat_in)
                loss += args.feature_scale * feature_loss
                data_logger["feature_loss"].append(feature_loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()

        # Log losses and other metrics for evaluation!
        data_logger["mu"].append(mu.mean().item())
        data_logger["mu_var"].append(mu.var().item())
        data_logger["log_var"].append(log_var.mean().item())
        data_logger["log_var_var"].append(log_var.var().item())

        data_logger["kl_loss"].append(kl_loss.item())
        data_logger["img_mse"].append(mse_loss.item())

        # Save results and a checkpoint at regular intervals

    print("---Train---", "kl_loss:", kl_loss.item(), "img_mse:", mse_loss.item())

    # In eval mode the model will use mu as the encoding instead of sampling from the distribution
    vae_net.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # Save an example from testing and log a test loss
            recon_img, mu, log_var = vae_net(test_images.to(device))
            loss = F.mse_loss(recon_img, gt_images.to(device)).item()

            data_logger['test_mse_loss'].append(loss)

            img_cat = torch.cat((recon_img.cpu(), test_images), 2).float()
            vutils.save_image(img_cat,
                              "%s/%s/%s_%d_test.png" % (args.save_dir,
                                                        "Results",
                                                        args.model_name,
                                                        args.image_size),
                              normalize=True)

        # Keep a copy of the previous save in case we accidentally save a model that has exploded...
        # if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
        #     shutil.copyfile(src=args.save_dir + "/Models/" + save_file_name + ".pt",
        #                     dst=args.save_dir + "/Models/" + save_file_name + "_copy.pt")

        # Save a checkpoint

        # Set the model back into training mode!!
        vae_net.train()

        print("---Test---", "img_mse: ", loss, "patience: ",patience)
        if loss < min_loss:
            min_loss = loss
            patience = 0
            torch.save({
                'epoch': epoch + 1,
                'data_logger': dict(data_logger),
                'model_state_dict': vae_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.save_dir + "/Models/" + save_file_name + f".pt")

        else:
            patience += 1
            if patience > threshold_patience:
                quit()