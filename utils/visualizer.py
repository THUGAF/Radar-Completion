import os
import torch
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors


REF_CMAP = pcolors.ListedColormap([[255 / 255, 255 / 255, 255 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                                   [10 / 255, 35 / 255, 244 / 255], [41 / 255, 253 / 255, 47 / 255], [30 / 255, 199 / 255, 34 / 255],
                                   [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                                   [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                                   [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255]])
REF_NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), REF_CMAP.N)

plt.rcParams['font.sans-serif'] = 'Arial'


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
            pm = ax.pcolormesh(thetas, rhos, tensors[r, c].T, cmap=REF_CMAP, norm=REF_NORM)
            if c == 0 or c == 1:
                ax.plot(np.ones(radial_size) * (anchor + azimuth_start_point) / 180 * np.pi,
                        np.arange(radial_start_point, radial_start_point + radial_size), 
                        '--', color='k', linewidth=1)
                ax.plot(np.ones(radial_size) * (anchor + blockage_len + azimuth_start_point) / 180 * np.pi,
                        np.arange(radial_start_point, radial_start_point + radial_size), 
                        '--', color='k', linewidth=1)
            if c == 0:
                ax.set_title(current_datetime[r], fontsize=10, loc='left', pad=1)
            ax.set_xlim(azimuth_start_point / 180 * np.pi, (azimuth_start_point + azimuth_size) / 180 * np.pi)
            ax.set_rlim(radial_start_point, radial_start_point + radial_size)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction('clockwise')
            # ax.set_theta_offset((azimuth_start_point + 180) / 180 * np.pi)
            ax.grid(True, linewidth=1)
            ax.tick_params(labelsize=10)

            cbar = fig.colorbar(pm, ax=ax, pad=0.2, aspect=20, shrink=0.4, extend='both')
            cbar.set_label('dBZ', fontsize=10)
            cbar.ax.tick_params(labelsize=8)

    fig.savefig('{}/{}.png'.format(root, stage), bbox_inches='tight')
    plt.close(fig)


def plot_psd(tensors: torch.Tensor, radial_start_point: float, anchor: int, blockage_len: int, 
             root: str, stage: str):
    tensors = tensors.detach().cpu()
    radial_size = tensors.size(3)
    pred, truth = tensors[-1, 0], tensors[-1, 1]
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

    fig = plt.figure(figsize=(15, 5), dpi=600)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(pred_radial_wavelength, pred_radial_mean_psd, color='b')
    ax1.plot(truth_radial_wavelength, truth_radial_mean_psd, color='r')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.invert_xaxis()
    ax1.set_xlabel('Wave length (km)', fontsize=12)
    ax1.set_ylabel('Radial Power Spectral Density', fontsize=12)
    ax1.legend(['Prediction', 'Groud truth'])

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(pred_azimuthal_wavelength, pred_azimuthal_mean_psd, color='b')
    ax2.plot(truth_azimuthal_wavelength, truth_azimuthal_mean_psd, color='r')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=10)
    ax2.invert_xaxis()
    ax2.set_xlabel('Wave length (deg)', fontsize=12)
    ax2.set_ylabel('Azimuthal Power Spectral Density', fontsize=12)
    ax2.legend(['Prediction', 'Groud truth'])

    fig.savefig('{}/{}_psd.png'.format(root, stage), bbox_inches='tight')
    plt.close(fig)
