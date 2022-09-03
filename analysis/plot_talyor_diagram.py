import os
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from taylor_diagram import TaylorDiagram
from typing import List


plt.rcParams['font.sans-serif'] = 'Arial'

def plot_taylor_diagram(root: str, paths: List, models: List, target_path: str, std_range: List, std_num: int, colors: List):
    fig = plt.figure(figsize=(10, 4), dpi=600)

    truth = torch.load(os.path.join(root, paths[-1], 'truth', 'truth.pt'))
    truth_30min, truth_60min = truth[4, 0, 0], truth[9, 0, 0]
    ref_std_30min, ref_std_60min = torch.std(truth_30min), torch.std(truth_60min)

    taylor_diagram_30min = TaylorDiagram(ref_std_30min.numpy(), fig, rect=121, 
                                         std_min=std_range[0], std_max=std_range[1],
                                         std_label_format='%.0f', num_std=std_num, 
                                         label='Observation', ylabel_text='$\sigma_{\hat{y}}$')
    taylor_diagram_60min = TaylorDiagram(ref_std_60min.numpy(), fig, rect=122, 
                                         std_min=std_range[0], std_max=std_range[1],
                                         std_label_format='%.0f', num_std=std_num, 
                                         label='Observation', ylabel_text='$\sigma_{\hat{y}}$')

    for i, path in enumerate(paths):
        pred = torch.load(os.path.join(root, path, 'pred', 'pred.pt'))
        pred_30min, pred_60min = pred[4, 0, 0], pred[9, 0, 0]
        stddev_30min, stddev_60min = torch.std(pred_30min), torch.std(pred_60min)
        corrcoef_30min = torch.corrcoef(torch.stack([truth_30min.flatten(), pred_30min.flatten()]))[0, 1]
        corrcoef_60min = torch.corrcoef(torch.stack([truth_60min.flatten(), pred_60min.flatten()]))[0, 1]
        taylor_diagram_30min.add_sample(stddev_30min.numpy(), corrcoef_30min.numpy(),
                                        marker='$%d$' % (i + 1), ms=5, ls='',
                                        mfc=colors[i], mec=colors[i], label=models[i])
        taylor_diagram_60min.add_sample(stddev_60min.numpy(), corrcoef_60min.numpy(),
                                        marker='$%d$' % (i + 1), ms=5, ls='',
                                        mfc=colors[i], mec=colors[i], label=models[i])
    
    # Add grid
    taylor_diagram_30min.add_grid()
    taylor_diagram_60min.add_grid()

    # Add RMS contours, and label them
    contours_30 = taylor_diagram_30min.add_contours(colors='0.5')
    plt.clabel(contours_30, inline=1, fontsize='medium', fmt='%.2f')
    contours_60 = taylor_diagram_60min.add_contours(colors='0.5')
    plt.clabel(contours_60, inline=1, fontsize='medium', fmt='%.2f')
    
    # Add a figure legend
    taylor_diagram_30min.ax.legend(taylor_diagram_30min.samplePoints,
                                   [p.get_label() for p in taylor_diagram_30min.samplePoints],
                                   numpoints=1, fontsize=8, bbox_to_anchor=(1.2, 1.1))
    taylor_diagram_60min.ax.legend(taylor_diagram_60min.samplePoints,
                                   [p.get_label() for p in taylor_diagram_60min.samplePoints],
                                   numpoints=1, fontsize=8, bbox_to_anchor=(1.2, 1.1))
    
    # Add title
    fig.axes[0].set_title('(a)\n', loc='left')
    fig.axes[1].set_title('(b)\n', loc='left')

    fig.tight_layout()
    fig.savefig(target_path)


if __name__ == '__main__':
    colors = cm.get_cmap('tab10')
    plot_taylor_diagram('results', 
                        ['AttnUNet/sample', 'AttnUNet_CV/sample', 'AttnUNet_GAN/sample', 'AttnUNet_GAN_CV/sample'], 
                        ['AGAN(g)', 'AGAN(g)+SVRE', 'AGAN', 'AGAN+SVRE'], 
                        'img/taylor_ablation.jpg', std_range=[7, 17], std_num=6, 
                        colors=colors.colors)
    plot_taylor_diagram('results',
                        ['PySTEPS/sample', 'ConvLSTM/sample', 'SmaAt_UNet/sample', 'AttnUNet_GAN_CV/sample'], 
                        ['PySTEPS', 'ConvLSTM-EF', 'SmaAt-UNet', 'AGAN+SVRE (ours)'],
                        'img/taylor_comparison.jpg', std_range=[7, 17], std_num=6, 
                        colors=colors.colors)
