import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors
import matplotlib.cm as cm
import sys
sys.path.append(os.getcwd())
import train


plt.rcParams['font.sans-serif'] = 'Arial'
CMAP = pcolors.ListedColormap([[255 / 255, 255 / 255, 255 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                               [10 / 255, 35 / 255, 244 / 255], [41 / 255, 253 / 255, 47 / 255], [30 / 255, 199 / 255, 34 / 255],
                               [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                               [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                               [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255]])
NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), CMAP.N)

AZIMUTH_START_POINT = 0
RADIAL_START_POINT = 0
ANCHOR = [180]
BLOCKAGE_LEN = [40]


def plot_refs(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    num_subplot = len(model_names) + 1
    fig = plt.figure(figsize=(num_subplot // 2 * 6, 12), dpi=600)
    
    truth = torch.load(os.path.join(model_dirs[0], '{}.pt'.format(stage)))[0, 0]
    azimuth_size, radial_size = truth.size(0), truth.size(1)
    thetas = np.arange(AZIMUTH_START_POINT, AZIMUTH_START_POINT + azimuth_size) / 180 * np.pi
    rhos = np.arange(RADIAL_START_POINT, RADIAL_START_POINT + radial_size)
    thetas, rhos = np.meshgrid(thetas, rhos)
    
    for i in range(num_subplot):
        if i == 0:
            tensor = truth
        else:
            tensor = torch.load(os.path.join(model_dirs[i - 1], '{}.pt'.format(stage)))[0, 1]
        ax = fig.add_subplot(2, num_subplot // 2, i + 1, projection='polar')
        title = 'Truth' if i == 0 else model_names[i - 1]
        pm = ax.pcolormesh(thetas, rhos, tensor.T, cmap=CMAP, norm=NORM)
        anchor, blockage_len = ANCHOR[int(stage[-1])], BLOCKAGE_LEN[int(stage[-1])]
        ax.plot(np.ones(radial_size) * (anchor + AZIMUTH_START_POINT) / 180 * np.pi,
                np.arange(RADIAL_START_POINT, RADIAL_START_POINT + radial_size), 
                '--', color='k', linewidth=1)
        ax.plot(np.ones(radial_size) * (anchor + blockage_len + AZIMUTH_START_POINT) / 180 * np.pi,
                np.arange(RADIAL_START_POINT, RADIAL_START_POINT + radial_size), 
                '--', color='k', linewidth=1)

        ax.set_title(title, fontsize=20, loc='left', pad=1)
        ax.set_xlim(AZIMUTH_START_POINT / 180 * np.pi, (AZIMUTH_START_POINT + azimuth_size) / 180 * np.pi)
        ax.set_rlim(RADIAL_START_POINT, RADIAL_START_POINT + radial_size)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction('clockwise')
        ax.grid(True, linewidth=1)
        ax.tick_params(labelsize=16)
    
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.94, 0.14, 0.012, 0.72])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax, orientation='vertical')
    cbar.set_label('dBZ', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


def plot_psd(model_names, model_dirs, stage, img_path_1, img_path_2):
    print('Plotting {} ...'.format(img_path_1))
    print('Plotting {} ...'.format(img_path_2))
    psd_df_radial = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_radial.csv'.format(stage)))
    psd_df_azimuthal = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_azimuthal.csv'.format(stage)))
    radial_wavelength, truth_psd_radial = psd_df_radial['wavelength_radial'], psd_df_radial['truth_psd_radial']
    azimuthal_wavelength, truth_psd_azimuthal = psd_df_azimuthal['wavelength_azimuthal'], psd_df_azimuthal['truth_psd_azimuthal']

    fig1 = plt.figure(figsize=(8, 4), dpi=600)
    fig2 = plt.figure(figsize=(8, 4), dpi=600)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    
    ax1.plot(radial_wavelength, truth_psd_radial, color='k')
    ax2.plot(azimuthal_wavelength, truth_psd_azimuthal, color='k')
    legend = ['Truth']
    for i in range(len(model_names)):
        psd_df_radial = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_radial.csv'.format(stage)))
        psd_df_azimuthal = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_azimuthal.csv'.format(stage)))
        pred_psd_radial, pred_psd_azimuthal = psd_df_radial['pred_psd_radial'], psd_df_azimuthal['pred_psd_azimuthal']
        ax1.plot(radial_wavelength, pred_psd_radial)
        ax2.plot(azimuthal_wavelength, pred_psd_azimuthal)
        legend.append(model_names[i])

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.invert_xaxis()
    ax1.set_xlabel('Wave Length (km)', fontsize=14)
    ax1.set_ylabel('Power spectral density of X axis', fontsize=14)
    ax1.legend(legend)

    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('Wave Length (km)', fontsize=14)
    ax2.set_ylabel('Power spectral density of Y axis', fontsize=14)
    ax2.legend(legend)

    fig1.savefig(img_path_1, bbox_inches='tight')
    fig2.savefig(img_path_2, bbox_inches='tight')
    print('{} saved'.format(img_path_1))
    print('{} saved'.format(img_path_2))


if __name__ == '__main__':
    model_names = ['Upper', 'GLCIC', 'UNetppL3', 'UNet', 'DilatedUNet']
    model_dirs = [os.path.join('results', m) for m in model_names]
    plot_refs(model_names, model_dirs, 'sample_0', 'results/ppi_sample_0.jpg')
    plot_psd(model_names, model_dirs, 'sample_0', 'results/psd_radial_sample_0.jpg', 'results/psd_azimuthal_sample_0.jpg')
