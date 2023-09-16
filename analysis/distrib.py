import os
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
def get_dataset(root: str):
    filenames = sorted(glob.glob(os.path.join(root, '*/*.npz')))
    ref = np.load(filenames[0])['ref'][1][:, :80]
    total_ref = np.zeros((len(filenames),) + ref.shape)
    for i, filename in enumerate(tqdm.tqdm(filenames)):
        ref = np.load(filename)['ref'][1][:, :80]
        ref = ref
        total_ref[i] = ref
    return total_ref


def draw_distrib(data: np.ndarray, img_path: str):
    print('Plotting {} ...'.format(img_path))
    fig = plt.figure(figsize=(8, 4), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    data = np.ravel(data)
    ax.hist(data, color='royalblue', bins=np.arange(0, 75, 5), 
            edgecolor='w', density=True, log=True)
    ax.hist(data, color='seagreen', bins=np.arange(-35, 5, 5), 
            edgecolor='w', density=True, log=True)
    ax.legend(['Retained', 'Clipped'], fontsize=10, edgecolor='w', fancybox=False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_xlim(left=-34, right=70)
    ax.set_ylim(top=1)
    ax.set_xlabel('Radar Reflectivity (dBZ)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.tick_params(labelsize=11)
    
    ax.axvline(x=0, color='k')
    ax.axvline(x=10, color='k')
    ax.axvline(x=20, color='k')
    ax.axvline(x=30, color='k')
    ax.axvline(x=40, color='k')
    ax.text(x=2, y=0.1, s='$w=1$', fontsize=10)
    ax.text(x=12, y=0.1, s='$w=2$', fontsize=10)
    ax.text(x=22, y=0.1, s='$w=5$', fontsize=10)
    ax.text(x=31, y=0.1, s='$w=10$', fontsize=10)
    ax.text(x=41, y=0.1, s='$w=30$', fontsize=10)


    fig.savefig(img_path, bbox_inches='tight')
    print('{} saved'.format(img_path))


if __name__ == '__main__':
    total_ref = get_dataset('/data/gaf/SBandRawNPZ')
    draw_distrib(total_ref, 'results/distrib.jpg')
