import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable





def vectorgraph0(data, save_path, sample_indices, dpi=600):
    num_samples = data.shape[0]
    layer1 = data[..., :3]  # First 3 channels
    layer2 = data[..., 3:]  # Last 3 channels
    fig, axs = plt.subplots(2, 10, dpi=dpi, figsize=(20, 5))
    for i, sample_idx in enumerate(sample_indices):
        arr1=layer1[sample_idx]
        arr2=layer2[sample_idx]
        axs[0, i].quiver(np.arange(arr1.shape[0]), np.arange(arr1.shape[1]), arr1[:, :, 0].T, arr1[:, :, 1].T, arr1[:, :, 2].T, clim=[-0.5, 0.5])
        axs[0, i].axis('off')
        axs[0, i].set_title('Sample {}'.format(sample_idx))
        axs[1, i].quiver(np.arange(arr1.shape[0]), np.arange(arr1.shape[0]), arr2[:, :, 0].T, arr2[:, :, 1].T, arr2[:, :, 2].T, clim=[-0.5, 0.5])
        axs[1, i].axis('off')
        axs[1, i].set_title('Sample {}'.format(sample_idx))
    plt.tight_layout()
    plt.savefig('{}.png'.format(save_path), dpi=dpi)
    plt.close()


def np2rgb(data, save_path, sample_indices, dpi=600):
    num_samples = data.shape[0]
    layer1 = data[..., :3]  # First 3 channels
    layer2 = data[..., 3:]  # Last 3 channels
    fig, axs = plt.subplots(2, 10, dpi=dpi, figsize=(20, 5))
    for i, sample_idx in enumerate(sample_indices):
        normalized_layer1 = (layer1[sample_idx] - layer1[sample_idx].min()) / (layer1[sample_idx].max() - layer1[sample_idx].min())
        normalized_layer2 = (layer2[sample_idx] - layer2[sample_idx].min()) / (layer2[sample_idx].max() - layer2[sample_idx].min())
        axs[0, i].imshow(normalized_layer1)
        axs[0, i].axis('off')
        axs[0, i].set_title('Sample {}'.format(sample_idx))
        axs[1, i].imshow(normalized_layer2)
        axs[1, i].axis('off')
        axs[1, i].set_title('Sample {}'.format(sample_idx))
    plt.tight_layout()
    plt.savefig('{}.png'.format(save_path), dpi=dpi)
    plt.close()



def plot_histograms(data, save_path, dpi=600):
    num_channels = data.shape[-1]
    num_bins = 50  # Number of bins in the histograms

    fig, axs = plt.subplots(2, 3, figsize=(16, 8), dpi=dpi)

    for i in range(num_channels):
        row = i // 3  # Determine which row (0 or 1) the histogram should go to
        col = i % 3   # Determine which column (0, 1, or 2) the histogram should go to

        channel_data = data[..., i].flatten()
        axs[row, col].hist(channel_data, bins=num_bins, color='blue', alpha=0.7)
        axs[row, col].set_title(f'Channel {i+1} Histogram')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()




if __name__ == "__main__":

    #Walk through the folder to check the data
    folder_path = "./data_Hd32/"
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    sample_indices = [0, 5, 10, 30, 50, 100, 200, 300, 500, 1000]

    # visualize spin case array
    for path in tqdm(subfolders):
        
        #draw spin
        print('Process: ', path + '/Spins.npy')
        data = np.load(path + '/Spins.npy')
        np2rgb(data, path+'/Spins_rgb', sample_indices)
        vectorgraph0(data, path+'/Spins_vector', sample_indices)
        plot_histograms(data, path+'/Spins_hist')
        
        #draw Hd
        print('Process: ', path + '/Hds.npy')
        data = np.load(path + '/Hds.npy')
        np2rgb(data, path+'/Hds_rgb', sample_indices)
        vectorgraph0(data, path+'/Hds_vector', sample_indices)
        plot_histograms(data, path+'/Hd_hist')
        
