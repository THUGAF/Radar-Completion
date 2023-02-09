import os
import numpy as np
import pandas as pd
import torch
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
LEFT_BOTTOM_LAT, LEFT_BOTTOM_LON = TRANS_UTM_TO_WGS84.transform(CENTER_UTM_X, CENTER_UTM_Y)
RIGHT_TOP_LAT, RIGHT_TOP_LON = TRANS_UTM_TO_WGS84.transform(CENTER_UTM_X, CENTER_UTM_Y)
AREA = [LEFT_BOTTOM_LON, RIGHT_TOP_LON, LEFT_BOTTOM_LAT, RIGHT_TOP_LAT]

CMAP = pcolors.ListedColormap([[255 / 255, 255 / 255, 255 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                                [10 / 255, 35 / 255, 244 / 255], [41 / 255, 253 / 255, 47 / 255], [30 / 255, 199 / 255, 34 / 255],
                                [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                                [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                                [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255]])
NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), CMAP.N)

def plot_refs(model_names, model_dirs, stage, img_path):
    print('Plotting {} ...'.format(img_path))
    tensor = torch.load(os.path.join(model_dirs[0], '{}/{}.pt'.format(stage, stage)))
    tensor = np.flip(tensor[0].numpy(), axis=0)
    
    num_subplot = len(model_names) + 1
    fig = plt.figure(figsize=(num_subplot // 2 * 6, 12), dpi=600)
    for i in range(num_subplot):
        ax = fig.add_subplot(2, num_subplot // 2, i + 1, projection=ccrs.Mercator())
        title = 'Truth' if i == 0 else model_names[i - 1]
        ax.set_extent(AREA, crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.STATES)
        ax.imshow(tensor, cmap=CMAP, norm=NORM, extent=AREA, transform=ccrs.PlateCarree())

        xticks = np.arange(np.ceil(2 * AREA[0]) / 2, np.ceil(2 * AREA[1]) / 2, 0.5)
        yticks = np.arange(np.ceil(2 * AREA[2]) / 2, np.ceil(2 * AREA[3]) / 2, 0.5)
        ax.set_xticks(np.arange(np.ceil(AREA[0]), np.ceil(AREA[1]), 1), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(np.ceil(AREA[2]), np.ceil(AREA[3]), 1), crs=ccrs.PlateCarree())
        ax.gridlines(crs=ccrs.PlateCarree(), xlocs=xticks, ylocs=yticks, draw_labels=False, 
                    linewidth=1, linestyle=':', color='k', alpha=0.8)

        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=18)
        ax.set_title(title, fontsize=22)
    
    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.94, 0.14, 0.012, 0.72])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), cax=cax, orientation='vertical')
    cbar.set_label('dBZ', fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


def plot_psd(model_names, model_dirs, stage, img_path_1, img_path_2):
    print('Plotting {} ...'.format(img_path_1))
    print('Plotting {} ...'.format(img_path_2))
    psd_radial_df = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_radial.csv'.format(stage)))
    psd_azimuthal_df = pd.read_csv(os.path.join(model_dirs[0], '{}_psd_azimuthal.csv'.format(stage)))
    radial_wavelength, truth_radial_psd = psd_radial_df['radial_wavelength'], psd_radial_df['truth_radial_psd']
    azimuthal_wavelength, truth_azimuthal_psd = psd_azimuthal_df['azimuthal_wavelength'], psd_azimuthal_df['truth_azimuthal_psd']

    fig1 = plt.figure(figsize=(8, 4), dpi=600)
    fig2 = plt.figure(figsize=(8, 4), dpi=600)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)

    ax1.plot(radial_wavelength, truth_radial_psd, color='k')
    ax2.plot(azimuthal_wavelength, truth_azimuthal_psd, color='k')

    legend = ['Truth']
    for i in range(len(model_names)):
        psd_radial_df = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_radial.csv'.format(stage)))
        psd_azimuthal_df = pd.read_csv(os.path.join(model_dirs[i], '{}_psd_azimuthal.csv'.format(stage)))
        pred_radial_psd, pred_azimuthal_psd = psd_radial_df['pred_psd_radial'], psd_azimuthal_df['pred_psd_azimuthal']
        ax1.plot(radial_wavelength, pred_radial_psd)
        ax2.plot(azimuthal_wavelength, pred_azimuthal_psd)
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
