import os
import datetime

import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors
import matplotlib.cm as cm

import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


plt.rcParams['font.sans-serif'] = 'Arial'

# Coordinate transformation
TRANS_WGS84_TO_UTM = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857')
TRANS_UTM_TO_WGS84 = pyproj.Transformer.from_crs('epsg:3857', 'epsg:4326')

# Global information
CENTER_LON, CENTER_LAT = 116.47195, 39.808887
CENTER_UTM_X, CENTER_UTM_Y = TRANS_WGS84_TO_UTM.transform(CENTER_LAT, CENTER_LON)
LEFT_BOTTOM_LAT, LEFT_BOTTOM_LON = TRANS_UTM_TO_WGS84.transform(CENTER_UTM_X - 128000, CENTER_UTM_Y - 64000)
RIGHT_TOP_LAT, RIGHT_TOP_LON = TRANS_UTM_TO_WGS84.transform(CENTER_UTM_X + 128000, CENTER_UTM_Y + 192000)
AREA = [LEFT_BOTTOM_LON, RIGHT_TOP_LON, LEFT_BOTTOM_LAT, RIGHT_TOP_LAT]

REF_CMAP = pcolors.ListedColormap([[255 / 255, 255 / 255, 255 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                                   [10 / 255, 35 / 255, 244 / 255], [41 / 255, 253 / 255, 47 / 255], [30 / 255, 199 / 255, 34 / 255],
                                   [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                                   [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                                   [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255]])
REF_NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), REF_CMAP.N)

PRCP_CMAP = pcolors.ListedColormap([[255 / 255, 255 / 255, 255 / 255], [204 / 255, 243 / 255, 203 / 255], [173 / 255, 234 / 255, 169 / 255], 
                                    [93 / 255, 190 / 255, 107 / 255], [120 / 255, 190 / 255, 252 / 255], [44 / 255, 56 / 255, 227 / 255], 
                                    [235 / 255, 61 / 255, 248 / 255], [215 / 255, 106 / 255, 62 / 255], [140 / 255, 48 / 255, 104 / 255]])
PRCP_NORM = pcolors.BoundaryNorm([0, 0.1, 1, 2, 5, 10, 20, 30, 50, 100], PRCP_CMAP.N)


def plot_loss(train_loss, val_loss, output_path, filename='loss.png'):
    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = plt.subplot(111)
    ax.plot(range(1, len(train_loss) + 1), train_loss, 'b')
    ax.plot(range(1, len(val_loss) + 1), val_loss, 'r')
    ax.set_xlabel('epoch')
    ax.legend(['train loss', 'val loss'])
    fig.savefig(os.path.join(output_path, filename), bbox_inches='tight')
    plt.close(fig)


def plot_map(input_, pred, truth, timestamp, root, stage):
    print('Plotting maps...')
    if not os.path.exists(os.path.join(root, stage)):
        os.mkdir(os.path.join(root, stage))
    _plot_map_figs(input_, root, timestamp[:input_.size(0)], stage, type='input', 
                   cmap=REF_CMAP, norm=REF_NORM)
    _plot_map_figs(pred, root, timestamp[input_.size(0): input_.size(0) + pred.size(0)], 
                   stage, type='pred', cmap=REF_CMAP, norm=REF_NORM)
    _plot_map_figs(truth, root, timestamp[input_.size(0): input_.size(0) + truth.size(0)], 
                   stage, type='truth', cmap=REF_CMAP, norm=REF_NORM)


def _plot_map_figs(tensor, root, timestamp, stage, type, cmap, norm):
    path = os.path.join(root, stage, type)
    if not os.path.exists(path):
        os.mkdir(path)

    # inverse scaling
    tensor = tensor.detach().cpu()

    image_list = []
    for i in range(tensor.size(0)):
        # minus represents the time before current moment
        if type == 'input':
            str_min = str(6 * (i - tensor.size(0) + 1))
        else:
            str_min = str(6 * (i + 1))
        file_path = '{}/{}.png'.format(path, str_min)
        current_datetime = datetime.datetime.utcfromtimestamp(int(timestamp[i, 0]))
        _plot_map_fig(tensor[i, 0, 0], file_path, current_datetime, cmap, norm)
        image_list.append(imageio.imread(file_path))

    # plot the long image
    num_rows = 2 if tensor.size(0) > 10 else 1
    num_cols = tensor.size(0) // num_rows
    fig = plt.figure(figsize=(num_cols, num_rows), dpi=300)
    for i in range(tensor.size(0)):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(np.flip(tensor[i, 0, 0].numpy(), axis=0), cmap=cmap, norm=norm)
        ax.axis('off')
    
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    fig.savefig('{}/{}.png'.format(path, type))
    plt.close(fig)
    
    # make gif
    imageio.mimsave('{}/{}.gif'.format(path, type), image_list, 'GIF', duration=0.2)

    # save tensor
    torch.save(tensor, '{}/{}.pt'.format(path, type))


def _plot_map_fig(tensor_slice, file_path, current_datetime, cmap, norm):
    fig = plt.figure(figsize=(8, 8), dpi=300)
    fig.suptitle('\n' + current_datetime.strftime('%Y-%m-%d %H:%M:%S'), fontsize=24)
    ax = plt.subplot(111, projection=ccrs.Mercator())
    ax.set_title('CR', fontsize=18)
    ax.set_extent(AREA, crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)

    tensor_slice = np.flip(tensor_slice.numpy(), axis=0)
    ax.imshow(tensor_slice, cmap=cmap, norm=norm, extent=AREA, transform=ccrs.PlateCarree())

    xticks = np.arange(np.ceil(2 * AREA[0]) / 2, np.ceil(2 * AREA[1]) / 2, 0.5)
    yticks = np.arange(np.ceil(2 * AREA[2]) / 2, np.ceil(2 * AREA[3]) / 2, 0.5)
    ax.set_xticks(np.arange(np.ceil(AREA[0]), np.ceil(AREA[1]), 1), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.ceil(AREA[2]), np.ceil(AREA[3]), 1), crs=ccrs.PlateCarree())
    ax.gridlines(crs=ccrs.PlateCarree(), xlocs=xticks, ylocs=yticks, draw_labels=False, 
                 linewidth=1, linestyle=':', color='k', alpha=0.8)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=18)

    cbar = fig.colorbar(cm.ScalarMappable(cmap=REF_CMAP, norm=REF_NORM), pad=0.05, shrink=0.7, aspect=20, 
                        orientation='vertical', extend='both')
    cbar.set_label('dBZ', fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    plt.subplots_adjust(left=0.1, right=1, bottom=0, top=1)
    fig.savefig(file_path)
    plt.close(fig)
