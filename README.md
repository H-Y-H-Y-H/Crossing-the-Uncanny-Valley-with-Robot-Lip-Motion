# Crossing the 'Uncanny Valley' with Robot Lip Motion

This repository contains the official implementation and data for our work on achieving realistic lip synchronization in humanoid robots. 

---
## Repository Structure

```bash
.
├── om/                    # Our main method and source code
│   ├── Latent/            # Comparison experiment data (raw + code)
│   ├── Latent2Cmds/       # "Facial Action Transformer" code (model architecture + deployment)
│   ├── train_vae.py       # program to train the VAE model
│   ├── RES_VAE_Dynamic.py # VAE model (two files used by the pipeline)
│   ├── vgg19.py           # vgg19 for VAE model
├── main.py                # Deploy the VAE model
├── wavform_bl/            # Baseline folder (audio-amplitude-based approach)
├── nn_bl/                 # Baseline folder (nearest-neighbor approach)
└── README.md
```

## Training Data
Two types of training data are provided in the submission system.

- Real Robot Data: 'real_data.zip'
- Synthesized Robot Data: 'synthesized_data.zip'