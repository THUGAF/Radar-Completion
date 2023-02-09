import os
import torch
import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors
import matplotlib.cm as cm


plt.rcParams['font.sans-serif'] = 'Arial'
CMAP = pcolors.ListedColormap([[255 / 255, 255 / 255, 255 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                                [10 / 255, 35 / 255, 244 / 255], [41 / 255, 253 / 255, 47 / 255], [30 / 255, 199 / 255, 34 / 255],
                                [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                                [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                                [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255]])
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


def plot_ref(tensors: torch.Tensor, current_datetime: str, azimuth_start_point: float, radial_start_point: float, 
             anchor: int, blockage_len: int, root: str, stage: str):
    print('Plotting tensors...')
    
    # save tensor
    tensors = tensors.detach().cpu()
    torch.save(tensors, '{}/{}.pt'.format(root, stage))

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
            if c == 0 or c == 1:
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
    cax = fig.add_axes([0.94, 0.2, 0.01, 0.6])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax, orientation='vertical', extend='both')
    cbar.set_label('dBZ', fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    fig.savefig('{}/{}.png'.format(root, stage), bbox_inches='tight')


def plot_psd(tensors: torch.Tensor, radial_start_point: float, anchor: int, blockage_len: int, 
             root: str, stage: str):
    tensors = tensors.detach().cpu()
    radial_size = tensors.size(3)
    pred, truth = tensors[0, 0].numpy(), tensors[0, 1].numpy()
    thetas = np.arange(anchor, anchor + blockage_len) / 180 * np.pi
    rhos = np.arange(radial_start_point, radial_start_point + radial_size)
    thetas, rhos = np.meshgrid(thetas, rhos)
    pred = pred[anchor: anchor + blockage_len, radial_start_point: radial_start_point + radial_size]
    truth = truth[anchor: anchor + blockage_len, radial_start_point: radial_start_point + radial_size]
    
    pred_radial_freq, pred_radial_psd = scipy.signal.welch(pred, nperseg=pred.shape[1], axis=1)
    pred_radial_freq, pred_radial_mean_psd = pred_radial_freq[1:], np.mean(pred_radial_psd, axis=0)[1:]
    pred_radial_wavelength = 1 / pred_radial_freq
    pred_azimuthal_freq, pred_azimuthal_psd = scipy.signal.welch(pred, nperseg=pred.shape[0], axis=0)
    pred_azimuthal_freq, pred_azimuthal_mean_psd = pred_azimuthal_freq[1:], np.mean(pred_azimuthal_psd, axis=1)[1:]
    pred_azimuthal_wavelength = 1 / pred_azimuthal_freq

    truth_radial_freq, truth_radial_psd = scipy.signal.welch(truth, nperseg=truth.shape[1], axis=1)
    truth_radial_freq, truth_radial_mean_psd = truth_radial_freq[1:], np.mean(truth_radial_psd, axis=0)[1:]
    truth_radial_wavelength = 1 / truth_radial_freq
    truth_azimuthal_freq, truth_azimuthal_psd = scipy.signal.welch(truth, nperseg=truth.shape[0], axis=0)
    truth_azimuthal_freq, truth_azimuthal_mean_psd = truth_azimuthal_freq[1:], np.mean(truth_azimuthal_psd, axis=1)[1:]
    truth_azimuthal_wavelength = 1 / truth_azimuthal_freq

    radial_data = {
        'radial_wavelength': truth_radial_wavelength,
        'pred_radial_psd': pred_radial_mean_psd,
        'truth_radial_psd': truth_radial_mean_psd,
    }
    azimuthal_data = {
        'azimuthal_wavelength': truth_azimuthal_wavelength,
        'pred_azimuthal_psd': pred_azimuthal_mean_psd,
        'truth_azimuthal_psd': truth_azimuthal_mean_psd
    }
    radial_df = pd.DataFrame(radial_data)
    azimuthal_df = pd.DataFrame(azimuthal_data)
    radial_df.to_csv('{}/{}_psd_radial.csv'.format(root, stage), float_format='%.8f', index=False)
    azimuthal_df.to_csv('{}/{}_psd_azimuthal.csv'.format(root, stage), float_format='%.8f', index=False)

    fig1 = plt.figure(figsize=(8, 4), dpi=600)
    fig2 = plt.figure(figsize=(8, 4), dpi=600)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(pred_radial_wavelength, pred_radial_mean_psd, color='b')
    ax1.plot(truth_radial_wavelength, truth_radial_mean_psd, color='r')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.invert_xaxis()
    ax1.set_xlabel('Wave length (km)', fontsize=14)
    ax1.set_ylabel('Radial Power Spectral Density', fontsize=14)
    ax1.legend(['Prediction', 'Truth'])

    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(pred_azimuthal_wavelength, pred_azimuthal_mean_psd, color='b')
    ax2.plot(truth_azimuthal_wavelength, truth_azimuthal_mean_psd, color='r')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('Wave length (deg)', fontsize=14)
    ax2.set_ylabel('Azimuthal Power Spectral Density', fontsize=14)
    ax2.legend(['Prediction', 'Truth'])

    fig1.savefig('{}/{}_radial_psd.png'.format(root, stage), bbox_inches='tight')
    fig2.savefig('{}/{}_azimuthal_psd.png'.format(root, stage), bbox_inches='tight')
