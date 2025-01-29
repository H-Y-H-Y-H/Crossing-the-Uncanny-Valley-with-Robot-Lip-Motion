import numpy as np

import matplotlib.pyplot as plt

dist0 = np.loadtxt(f'dist_om_0.csv')
mean0 = np.mean(dist0)
std0 = np.std(dist0)

all_means = [mean0]
all_stds = [std0]

for i in range(11, 22):
    dist = np.loadtxt(f'dist_om_{i}.csv')
    all_means.append(np.mean(dist))
    all_stds.append(np.std(dist))


# Custom x-axis labels
labels = ['English1', 'French', 'Japanese', 'Kearo', 'English2',
          'Spanish', 'Italian', 'German', 'Russian', 'Chinese', 'Hebrew','Arabic']
plt.figure(figsize=(16,6))
# Plot with error bars
plt.bar(labels, all_means, yerr=all_stds, capsize=5, alpha=0.7,color="#069AF3")
plt.xlabel('Language')
plt.ylabel('Mean Latent Distance (Error)')
plt.title('Multilingual Lip Synchronization Error Analysis')
# plt.xticks(rotation=45)
# plt.tight_layout()  # Adjust layout to fit labels properly
plt.show()