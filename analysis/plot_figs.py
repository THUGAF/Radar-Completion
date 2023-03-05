import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors
import matplotlib.cm as cm


plt.rcParams['font.sans-serif'] = 'Arial'
CMAP = pcolors.ListedColormap(['#ffffff', '#2aedef', '#1caff4', '#0a22f4', '#29fd2f',
                               '#1ec722', '#139116', '#fffd38', '#e7bf2a', '#fb9124',
                               '#f90f1c', '#d00b15', '#bd0713', '#da66fb', '#bb24eb'])
NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), CMAP.N)

AZIMUTH_START_POINT = 0
RADIAL_START_POINT = 0
ANCHOR = [160]
BLOCKAGE_LEN = [40]


def plot_ppis(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    num_subplot = len(model_names) + 1
    fig = plt.figure(figsize=(num_subplot * 6, 6), dpi=600)
    
    truth = torch.load(os.path.join(model_dirs[0], '{}.pt'.format(stage)))[0, 1]
    azimuth_size, radial_size = truth.size(0), truth.size(1)
    thetas = np.arange(AZIMUTH_START_POINT, AZIMUTH_START_POINT + azimuth_size) / 180 * np.pi
    rhos = np.arange(RADIAL_START_POINT, RADIAL_START_POINT + radial_size)
    thetas, rhos = np.meshgrid(thetas, rhos)
    
    for i in range(num_subplot):
        if i == 0:
            tensor = truth
        else:
            tensor = torch.load(os.path.join(model_dirs[i - 1], '{}.pt'.format(stage)))[0, 0]
        ax = fig.add_subplot(1, num_subplot, i + 1, projection='polar')
        title = 'Truth' if i == 0 else model_names[i - 1]
        ax.grid(False)
        ax.pcolormesh(thetas, rhos, tensor.T, cmap=CMAP, norm=NORM)
        anchor, blockage_len = ANCHOR[int(stage[-1])], BLOCKAGE_LEN[int(stage[-1])]
        ax.plot(np.ones(radial_size) * (anchor + AZIMUTH_START_POINT) / 180 * np.pi, 
                np.arange(RADIAL_START_POINT, RADIAL_START_POINT + radial_size), 
                '--', color='k', linewidth=1)
        ax.plot(np.ones(radial_size) * (anchor + blockage_len + AZIMUTH_START_POINT) / 180 * np.pi, 
                np.arange(RADIAL_START_POINT, RADIAL_START_POINT + radial_size), 
                '--', color='k', linewidth=1)

        ax.set_title(title, loc='left', fontdict={'size': 20, 'weight': 'bold'})
        ax.set_xlim(AZIMUTH_START_POINT / 180 * np.pi, (AZIMUTH_START_POINT + azimuth_size) / 180 * np.pi)
        ax.set_rlim(RADIAL_START_POINT, RADIAL_START_POINT + radial_size)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction('clockwise')
        ax.grid(True, linewidth=1)
        ax.tick_params(labelsize=16)
    
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.14, 0.008, 0.72])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax, orientation='vertical')
    cbar.set_label('dBZ', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


def plot_psd(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    psd_df_radial = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_radial.csv'.format(stage)))
    psd_df_azimuthal = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_azimuthal.csv'.format(stage)))
    wavelength_radial, truth_psd_radial = psd_df_radial['wavelength_radial'], psd_df_radial['truth_psd_radial']
    wavelength_azimuthal, truth_psd_azimuthal = psd_df_azimuthal['wavelength_azimuthal'], psd_df_azimuthal['truth_psd_azimuthal']

    fig = plt.figure(figsize=(8, 8), dpi=600)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    ax1.plot(wavelength_radial, truth_psd_radial, color='k')
    ax2.plot(wavelength_azimuthal, truth_psd_azimuthal, color='k')
    legend = ['Truth']
    for i in range(len(model_names)):
        psd_df_radial = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_radial.csv'.format(stage)))
        psd_df_azimuthal = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_azimuthal.csv'.format(stage)))
        pred_psd_radial, pred_psd_azimuthal = psd_df_radial['pred_psd_radial'], psd_df_azimuthal['pred_psd_azimuthal']
        ax1.plot(wavelength_radial, pred_psd_radial)
        ax2.plot(wavelength_azimuthal, pred_psd_azimuthal)
        legend.append(model_names[i])

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.invert_xaxis()
    ax1.set_xlabel('Wave Length (km)', fontsize='large')
    ax1.set_ylabel('Radial power spectral density', fontsize='large')
    ax1.legend(legend)

    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('Wave Length (deg)', fontsize='large')
    ax2.set_ylabel('Azimuthal power spectral density', fontsize='large')
    ax2.legend(legend)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


def plot_radar_polygon(model_names, model_dirs, img_path):
    print('Plotting {} ...'.format(img_path))
    fig = plt.figure(figsize=(6, 6), dpi=600)
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    handles = []
    for i in range(len(model_names)):
        test_df = pd.read_csv(os.path.join(model_dirs[i], 'test_metrics.csv'))
        test_metrics = test_df.iloc[-1].values
        test_metrics[0], test_metrics[1] = test_metrics[0] / 10, test_metrics[1] / 10
        sample_df = pd.read_csv(os.path.join(model_dirs[i], 'sample_0_metrics.csv'))
        sample_metrics = sample_df.iloc[-1].values
        sample_metrics[0], sample_metrics[1] = sample_metrics[0] / 10, sample_metrics[1] / 10
        metrics = np.concatenate([test_metrics, sample_metrics])
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        labels = np.append(['${}_{{test}}$'.format(j) for j in test_df.columns],
                           ['${}_{{sample}}$'.format(j) for j in sample_df.columns])
        metrics = np.append(metrics, metrics[0])
        angles = np.append(angles, angles[0])
        labels = np.append(labels, labels[0])
        h, = ax.plot(angles, metrics)
        for j in np.arange(0, 1.4, 0.2):
            ax.plot(angles, [j] * len(angles), '-.', lw=0.5, color='black')
        for angle in angles:
            ax.plot([angle, angle], [0, 1.2], '-.', lw=0.5, color='black')
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        ax.set_theta_zero_location('N')
        ax.set_rlabel_position(0)
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        handles.append(h)
    
    fig.legend(labels=model_names, handles=handles)
    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


if __name__ == '__main__':
    model_names = ['Upper', 'GLCIC', 'UNet++ GAN', 'Dilated UNet']
    model_dirs = ['results/Upper', 'results/GLCIC_GAN', 'results/UNetpp_GAN', 'results/DilatedUNet']
    plot_ppis(model_names, model_dirs, 'sample_0', 'results/ppi_sample_0.jpg')
    plot_psd(model_names, model_dirs, 'sample_0', 'results/psd_sample_0.jpg')
