import os
import torch
import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors
import matplotlib.cm as cm


plt.rcParams['font.sans-serif'] = 'Arial'
CMAP = pcolors.ListedColormap(['#ffffff', '#2aedef', '#1caff4', '#0a22f4', '#29fd2f',
                               '#1ec722', '#139116', '#fffd38', '#e7bf2a', '#fb9124',
                               '#f90f1c', '#d00b15', '#bd0713', '#da66fb', '#bb24eb'])
NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), CMAP.N)


def plot_loss(train_loss: np.ndarray, val_loss: np.ndarray, output_path: str, filename: str = 'loss.png'):
    fig = plt.figure(figsize=(6, 4), dpi=600)
    ax = plt.subplot(111)
    ax.plot(range(1, len(train_loss) + 1), train_loss, 'b')
    ax.plot(range(1, len(val_loss) + 1), val_loss, 'r')
    ax.set_xlabel('epoch')
    ax.legend(['train loss', 'val loss'])
    fig.savefig(os.path.join(output_path, filename), bbox_inches='tight')
    plt.close(fig)


def save_tensor(tensor: torch.Tensor, root: str, stage: str):
    # save tensor
    tensor = tensor.detach().cpu()
    torch.save(tensor, '{}/{}.pt'.format(root, stage))


def plot_ppi(tensors: torch.Tensor, current_datetime: str, azimuth_start_point: float, radial_start_point: float, 
             anchor: int, blockage_len: int, root: str, stage: str):
    print('Plotting tensors...')
    # plot the long image
    num_rows, num_cols = tensors.size(0), tensors.size(1)
    azimuth_size, radial_size = tensors.size(2), tensors.size(3)
    thetas = np.arange(azimuth_start_point, azimuth_start_point + azimuth_size) / 180 * np.pi
    rhos = np.arange(radial_start_point, radial_start_point + radial_size)
    thetas, rhos = np.meshgrid(thetas, rhos)
    fig = plt.figure(figsize=(num_cols * 4, num_rows * 4), dpi=600)
    for r in range(num_rows):
        for c in range(num_cols):
            ax = fig.add_subplot(num_rows, num_cols, r * num_cols + c + 1, projection='polar')
            pm = ax.pcolormesh(thetas, rhos, tensors[r, c].T, cmap=CMAP, norm=NORM)
            ax.plot(np.ones(radial_size) * (anchor + azimuth_start_point) / 180 * np.pi,
                    np.arange(radial_start_point, radial_start_point + radial_size), 
                    '--', color='k', linewidth=1)
            ax.plot(np.ones(radial_size) * (anchor + blockage_len + azimuth_start_point) / 180 * np.pi,
                    np.arange(radial_start_point, radial_start_point + radial_size), 
                    '--', color='k', linewidth=1)
            if c == 0:
                ax.set_title(current_datetime[r], fontsize=12, loc='left', pad=1)
            ax.set_xlim(azimuth_start_point / 180 * np.pi, (azimuth_start_point + azimuth_size) / 180 * np.pi)
            ax.set_rlim(radial_start_point, radial_start_point + radial_size)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction('clockwise')
            ax.grid(True, linewidth=1)
            ax.tick_params(labelsize=12)

    fig.subplots_adjust(right=0.9, wspace=0.3, hspace=0.3)
    cax = fig.add_axes([0.95, 0.2, 0.005 * num_rows, 0.6])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax, orientation='vertical', extend='both')
    cbar.set_label('dBZ', fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    fig.savefig('{}/{}.png'.format(root, stage), bbox_inches='tight')


def plot_psd(tensors: torch.Tensor, radial_start_point: float, anchor: int, blockage_len: int, root: str, stage: str):
    tensors = tensors.detach().cpu()
    radial_size = tensors.size(3)
    pred, truth = tensors[0, 0].numpy(), tensors[0, 1].numpy()
    thetas = np.arange(anchor, anchor + blockage_len) / 180 * np.pi
    rhos = np.arange(radial_start_point, radial_start_point + radial_size)
    thetas, rhos = np.meshgrid(thetas, rhos)
    pred = pred[anchor: anchor + blockage_len, radial_start_point: radial_start_point + radial_size]
    truth = truth[anchor: anchor + blockage_len, radial_start_point: radial_start_point + radial_size]
    
    pred_freq_radial, pred_psd_radial = scipy.signal.welch(pred, nperseg=pred.shape[1], axis=1)
    pred_freq_radial, pred_psd_radial_mean = pred_freq_radial[1:], np.mean(pred_psd_radial, axis=0)[1:]
    pred_wavelength_radial = 1 / pred_freq_radial
    pred_freq_azimuthal, pred_psd_azimuthal = scipy.signal.welch(pred, nperseg=pred.shape[0], axis=0)
    pred_freq_azimuthal, pred_psd_azimuthal_mean = pred_freq_azimuthal[1:], np.mean(pred_psd_azimuthal, axis=1)[1:]
    pred_wavelength_azimuthal = 1 / pred_freq_azimuthal

    truth_freq_radial, truth_psd_radial = scipy.signal.welch(truth, nperseg=truth.shape[1], axis=1)
    truth_freq_radial, truth_radial_mean_psd = truth_freq_radial[1:], np.mean(truth_psd_radial, axis=0)[1:]
    truth_wavelength_radial = 1 / truth_freq_radial
    truth_freq_azimuthal, truth_psd_azimuthal = scipy.signal.welch(truth, nperseg=truth.shape[0], axis=0)
    truth_freq_azimuthal, truth_psd_azimuthal_mean = truth_freq_azimuthal[1:], np.mean(truth_psd_azimuthal, axis=1)[1:]
    truth_wavelength_azimuthal = 1 / truth_freq_azimuthal

    radial_data = {
        'wavelength_radial': truth_wavelength_radial,
        'pred_psd_radial': pred_psd_radial_mean,
        'truth_psd_radial': truth_radial_mean_psd,
    }
    azimuthal_data = {
        'wavelength_azimuthal': truth_wavelength_azimuthal,
        'pred_psd_azimuthal': pred_psd_azimuthal_mean,
        'truth_psd_azimuthal': truth_psd_azimuthal_mean
    }
    df_radial = pd.DataFrame(radial_data)
    df_azimuthal = pd.DataFrame(azimuthal_data)
    df_radial.to_csv('{}/{}_psd_radial.csv'.format(root, stage), float_format='%.8f', index=False)
    df_azimuthal.to_csv('{}/{}_psd_azimuthal.csv'.format(root, stage), float_format='%.8f', index=False)

    fig1 = plt.figure(figsize=(8, 4), dpi=600)
    fig2 = plt.figure(figsize=(8, 4), dpi=600)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(pred_wavelength_radial, pred_psd_radial_mean, color='b')
    ax1.plot(truth_wavelength_radial, truth_radial_mean_psd, color='r')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.invert_xaxis()
    ax1.set_xlabel('Wave length (km)', fontsize=14)
    ax1.set_ylabel('Radial Power Spectral Density', fontsize=14)
    ax1.legend(['Prediction', 'Truth'])

    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(pred_wavelength_azimuthal, pred_psd_azimuthal_mean, color='b')
    ax2.plot(truth_wavelength_azimuthal, truth_psd_azimuthal_mean, color='r')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('Wave length (deg)', fontsize=14)
    ax2.set_ylabel('Azimuthal Power Spectral Density', fontsize=14)
    ax2.legend(['Prediction', 'Truth'])

    fig1.savefig('{}/{}_psd_radial.png'.format(root, stage), bbox_inches='tight')
    fig2.savefig('{}/{}_psd_azimuthal.png'.format(root, stage), bbox_inches='tight')
