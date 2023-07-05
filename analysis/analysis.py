import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde


plt.rcParams['font.sans-serif'] = 'Arial'
CMAP = pcolors.ListedColormap(['#ffffff', '#2aedef', '#1caff4', '#0a22f4', '#29fd2f',
                               '#1ec722', '#139116', '#fffd38', '#e7bf2a', '#fb9124',
                               '#f90f1c', '#d00b15', '#bd0713', '#da66fb', '#bb24eb'])
NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), CMAP.N)

AZIMUTH_START_POINT = 0
RADIAL_START_POINT = 0
ANCHOR = [270, 40]
BLOCKAGE_LEN = [40, 40]


def plot_ppis(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    num_subplot = len(model_names) + 1
    num_row = (num_subplot + 1) // 2
    fig = plt.figure(figsize=(num_row * 6, 6 * 2), dpi=300)
    
    truth = torch.load(os.path.join(model_dirs[0], '{}.pt'.format(stage)))[0, 1]
    azimuth_size, radial_size = truth.size(0), truth.size(1)
    thetas = np.arange(AZIMUTH_START_POINT, AZIMUTH_START_POINT + azimuth_size) / 180 * np.pi
    rhos = np.arange(RADIAL_START_POINT, RADIAL_START_POINT + radial_size)
    thetas, rhos = np.meshgrid(thetas, rhos)
    anchor, blockage_len = ANCHOR[int(stage[-1])], BLOCKAGE_LEN[int(stage[-1])]
    truth = np.clip(truth, a_min=0, a_max=70)
    
    for i in range(num_subplot):
        if i == 0:
            tensor = truth
        else:
            tensor = torch.load(os.path.join(model_dirs[i - 1], '{}.pt'.format(stage)))[0, 0]
        ax = fig.add_subplot(2, num_row, i + 1, projection='polar')
        title = 'Observation' if i == 0 else model_names[i - 1]
        ax.grid(False)
        ax.pcolormesh(thetas, rhos, tensor.T, cmap=CMAP, norm=NORM)
        ax.plot(np.ones(radial_size) * (anchor + AZIMUTH_START_POINT) / 180 * np.pi, 
                np.arange(RADIAL_START_POINT, RADIAL_START_POINT + radial_size), 
                '--', color='k', linewidth=1)
        ax.plot(np.ones(radial_size) * (anchor + blockage_len + AZIMUTH_START_POINT) / 180 * np.pi, 
                np.arange(RADIAL_START_POINT, RADIAL_START_POINT + radial_size), 
                '--', color='k', linewidth=1)

        ax.set_title('({})'.format(format(chr(97 + i))), loc='left', fontsize=20)
        ax.set_title(title, loc='center', fontsize=20)
        ax.set_xlim(AZIMUTH_START_POINT / 180 * np.pi, (AZIMUTH_START_POINT + azimuth_size) / 180 * np.pi)
        ax.set_rlim(RADIAL_START_POINT, RADIAL_START_POINT + radial_size)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction('clockwise')
        ax.grid(True, linewidth=1)
        ax.tick_params(labelsize=16)
    
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.94, 0.14, 0.01, 0.72])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax, orientation='vertical')
    cbar.set_label('dBZ', fontsize=20, labelpad=20)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


def plot_css(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    num_subplot = len(model_names) + 1
    num_row = (num_subplot + 1) // 2
    fig = plt.figure(figsize=(num_row * 4, 4 * 2), dpi=300)

    truth = torch.load(os.path.join(model_dirs[0], '{}.pt'.format(stage)))[0, 1]
    azimuth_size, radial_size = truth.size(0), truth.size(1)
    thetas = np.arange(AZIMUTH_START_POINT, AZIMUTH_START_POINT + azimuth_size) / 180 * np.pi
    rhos = np.arange(RADIAL_START_POINT, RADIAL_START_POINT + radial_size)
    thetas, rhos = np.meshgrid(thetas, rhos)
    anchor, blockage_len = ANCHOR[int(stage[-1])], BLOCKAGE_LEN[int(stage[-1])]
    xs = truth[AZIMUTH_START_POINT + anchor: AZIMUTH_START_POINT + anchor + blockage_len, 
               RADIAL_START_POINT: RADIAL_START_POINT + radial_size]
    xs = xs.numpy().flatten()
    xs = np.clip(xs, a_min=0, a_max=70)

    for i in range(num_subplot - 1):
        pred = torch.load(os.path.join(model_dirs[i], '{}.pt'.format(stage)))[0, 0]
        ys = pred[AZIMUTH_START_POINT + anchor: AZIMUTH_START_POINT + anchor + blockage_len,
                  RADIAL_START_POINT: RADIAL_START_POINT + radial_size]
        ys = ys.numpy().flatten()
        ys = np.clip(ys, a_min=0, a_max=70)
        data = np.vstack([xs, ys])
        kde = gaussian_kde(data)
        density = kde.evaluate(data)
        ax = fig.add_subplot(2, num_row, i + 1)
        sc = ax.scatter(xs, ys, c=density, s=10, cmap='jet', norm=pcolors.Normalize(0, np.max(density)))
        
        ax.set_xlim([0, 60])
        ax.set_ylim([0, 60])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.axline((0, 0), (1, 1), color='k', linewidth=1, transform=ax.transAxes)
        ax.set_aspect('equal')
        ax.set_title(model_names[i], loc='center', fontsize=14, y=0.9)
        ax.set_title(' ({})'.format(format(chr(97 + i))), fontsize=14, loc='left', y=0.9)
        ax.set_xlabel('Observation (dBZ)', fontsize=14)
        if i == 0 or i == 3:
            ax.set_ylabel('Prediction (dBZ)', fontsize=14, labelpad=10)
        ax.tick_params(labelsize=12)
        
    cax = fig.add_subplot(2, num_row, num_subplot)
    cax.set_position([cax.get_position().x0, cax.get_position().y0,
                      cax.get_position().width * 0.1, cax.get_position().height])
    cbar = fig.colorbar(sc, cax=cax, orientation='vertical')
    cbar.set_label('Probability Density', fontsize=14, labelpad=20)
    cbar.ax.tick_params(labelsize=12)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))
    plt.close(fig)


def plot_psd(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    psd_df_radial = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_radial.csv'.format(stage)))
    psd_df_azimuthal = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_azimuthal.csv'.format(stage)))
    wavelength_radial, truth_psd_radial = psd_df_radial['wavelength_radial'], psd_df_radial['truth_psd_radial']
    wavelength_azimuthal, truth_psd_azimuthal = psd_df_azimuthal['wavelength_azimuthal'], psd_df_azimuthal['truth_psd_azimuthal']

    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    ax1.plot(wavelength_radial, truth_psd_radial, color='k')
    ax2.plot(wavelength_azimuthal, truth_psd_azimuthal, color='k')
    legend = ['Observation']
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
    ax1.legend(legend, loc='lower left', fontsize='small', edgecolor='w', fancybox=False)
    ax1.text(-0.1, 1.05, '(a)', fontsize=18, transform=ax1.transAxes)

    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('Wave Length (deg)', fontsize='large')
    ax2.set_ylabel('Azimuthal power spectral density', fontsize='large')
    ax2.legend(legend, loc='lower left', fontsize='small', edgecolor='w', fancybox=False)
    ax2.text(-0.1, 1.05, '(b)', fontsize=18, transform=ax2.transAxes)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


def plot_bars(model_names: list, model_dirs: list, stage: str, img_path: str):
    print('Plotting {} ...'.format(img_path))
    metrics = []
    num_models = len(model_names)
    for i in range(num_models):
        df = pd.read_csv(os.path.join(model_dirs[i], '{}_metrics.csv'.format(stage)), index_col=0)
        metrics.append(df.values)
    metrics = np.stack(metrics).transpose(0, 2, 1)

    num_subplot = len(df.columns)
    fig = plt.figure(figsize=(10, num_subplot * 4), dpi=300)
    for i in range(num_subplot):
        ax = fig.add_subplot(num_subplot, 1, i + 1)
        labels = df.index
        labels = ['{}~{}'.format(df.index[i], df.index[i + 1]) 
                  for i in range(len(df.index) - 2)] + ['>{}'.format(df.index[-2])] + [df.index[-1]]
        
        if i == num_subplot - 1:
            ax.set_xlabel('Radar Reflectivity (dBZ)', labelpad=5, fontsize=14)
        ax.set_ylabel(df.columns.values[i] + ' (dBZ)', labelpad=10, fontsize=14)
        x = np.arange(len(df.index))
        width = 0.18
        for j in range(num_models):
            b = ax.bar((x + width * (j - (num_models - 1) / 2)), metrics[j, i], width,
                   label=model_names[j], color=plt.get_cmap('Set3').colors[j], edgecolor='k')
            ax.bar_label(b, fmt='%.1f', padding=0.5, fontsize=9)
        ax.set_xticks(x, labels=labels)
        ax.axhline(color='k', linestyle='--', linewidth=1)
        ax.tick_params(labelsize=12)
        ax.text(-0.1, 1.05, '({})'.format(chr(97 + i)), fontsize=18, transform=ax.transAxes)

    fig.subplots_adjust(bottom=0.10)
    lax = fig.add_axes([0.1, 0, 0.8, 0.05])
    lax.set_axis_off()
    lax.legend(ax.containers, model_names, fontsize=12, loc='center', ncols=len(model_names),
               edgecolor='w', fancybox=False)
    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


def save_metrics(model_names: list, model_dirs: list, stage: str, file_path: str):
    print('Saving {} ...'.format(file_path))
    metrics = []
    num_models = len(model_names)
    for i in range(num_models):
        df = pd.read_csv(os.path.join(model_dirs[i], '{}_metrics.csv'.format(stage)), index_col=0)
        metrics.append(df.values)
    metrics = np.stack(metrics).transpose(0, 2, 1)
    writer = pd.ExcelWriter(file_path, mode='w')
    for i, idx in enumerate(df.columns):
        new_df = pd.DataFrame(metrics[:, i], index=model_names, columns=df.index)
        new_df.to_excel(writer, sheet_name=idx, float_format='%.4f')
    writer.close()
    print('{} saved'.format(file_path))


if __name__ == '__main__':
    model_names = ['MLG', 'BI', 'GLCIC', 'UNet++ GAN', 'DSA-UNet (Ours)']
    model_dirs = ['results/MLG', 'results/Bilinear', 'results/GLCIC', 'results/UNetpp_GAN', 'results/DSA_UNet']
    stages = ['test', 'case_0', 'case_1']
    for stage in stages:
        save_metrics(model_names, model_dirs, stage, 'results/img/{}_metrics.xlsx'.format(stage))
        plot_bars(model_names, model_dirs, stage, 'results/img/bar_{}.jpg'.format(stage))
        if stage != 'test':
            plot_ppis(model_names, model_dirs, stage, 'results/img/ppi_{}.jpg'.format(stage))
            plot_css(model_names, model_dirs, stage, 'results/img/cs_{}.jpg'.format(stage))
            plot_psd(model_names, model_dirs, stage, 'results/img/psd_{}.jpg'.format(stage))
    