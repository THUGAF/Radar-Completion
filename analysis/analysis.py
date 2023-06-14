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
    num_rows = (num_subplot + 1) // 2
    fig = plt.figure(figsize=(num_rows * 6, 6 * 2), dpi=300)
    
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
        ax = fig.add_subplot(2, num_rows, i + 1, projection='polar')
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
    cax = fig.add_axes([0.94, 0.14, 0.01, 0.72])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax, orientation='vertical')
    cbar.set_label('dBZ', fontsize=20, labelpad=20)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


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
    metrics = np.stack(metrics)

    num_subplot = len(df.index)
    num_rows = (num_subplot + 1) // 2
    fig = plt.figure(figsize=(2 * 6, num_rows * 4), dpi=300)
    for i in range(num_subplot):
        ax = fig.add_subplot((num_subplot + 1) // 2, 2, i + 1)
        if i < num_subplot - 2:
            title = '{}~{} dBZ'.format(df.index[i], df.index[i + 1])
        elif i == num_subplot - 2:
            title = '>{} dBZ'.format(df.index[i])
        else:
            title = 'total'
        ax.set_title(title, loc='right', fontsize=14)
        x = np.arange(len(df.columns))
        width = 0.16
        for j in range(num_models):
            ax.bar((x + width * (j - (num_models - 1) / 2)), metrics[j, i], width,
                   label=model_names[j], color=plt.get_cmap('Set3').colors[j], edgecolor='k')
        ax.axhline(color='k', linestyle='--', linewidth=1)
        ax.set_xticks(x, labels=df.columns.values)
        ax.tick_params(labelsize=12)
        ax.set_ylabel('Error (dBZ)', fontsize=12)
        ax.text(-0.1, 1.05, '({})'.format(chr(97 + i)), fontsize=18, transform=ax.transAxes)

    fig.subplots_adjust(bottom=0.06)
    lax = fig.add_axes([0, 0, 1, 0.04])
    lax.set_axis_off()
    lax.legend(ax.containers, model_names, fontsize=14, loc='center', ncols=len(model_names),
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
    metrics = np.stack(metrics)
    writer = pd.ExcelWriter(file_path, mode='w')
    for i, idx in enumerate(df.index):
        new_df = pd.DataFrame(metrics[:, i], index=model_names, columns=df.columns)
        new_df.to_excel(writer, sheet_name=idx, float_format='%.4f')
    writer.close()
    print('{} saved'.format(file_path))


if __name__ == '__main__':
    model_names = ['MLG', 'BI', 'GLCIC GAN', 'UNet++ GAN', 'DSA-UNet (Ours)']
    model_dirs = ['results/MLG', 'results/Bilinear', 'results/GLCIC', 'results/UNetpp_GAN', 'results/DSA_UNet']
    save_metrics(model_names, model_dirs, 'case_0', 'results/case_0_metrics.xlsx')
    save_metrics(model_names, model_dirs, 'test', 'results/test_metrics.xlsx')
    plot_ppis(model_names, model_dirs, 'case_0', 'results/ppi_case_0.jpg')
    plot_bars(model_names, model_dirs, 'case_0', 'results/bar_case_0.jpg')
    plot_bars(model_names, model_dirs, 'test', 'results/bar_test.jpg')
    plot_psd(model_names, model_dirs, 'case_0', 'results/psd_case_0.jpg')
    