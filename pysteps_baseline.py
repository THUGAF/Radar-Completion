import os
import argparse
import warnings

import torch
import pandas as pd
import numpy as np
import pysteps

from utils.dataloader import SampleDataset
import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.scaler as scaler


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='PySTEPS Basline')

# global settings
parser.add_argument('--data-path', type=str, default='/data/gaf/SBandCRUnzip')
parser.add_argument('--output-path', type=str, default='results/PySTEPS')
parser.add_argument('--sample-index', type=int, default=16840)
parser.add_argument('--lon-range', type=int, nargs='+', default=[271, 527])
parser.add_argument('--lat-range', type=int, nargs='+', default=[335, 591])
parser.add_argument('--seed', type=int, default=2021)

# input and output settings
parser.add_argument('--input-steps', type=int, default=10)
parser.add_argument('--forecast-steps', type=int, default=10)

# evaluation settings
parser.add_argument('--thresholds', type=int, nargs='+', default=[10, 15, 20, 25, 30, 35, 40])
parser.add_argument('--vmax', type=float, default=80.0)
parser.add_argument('--vmin', type=float, default=0.0)

args = parser.parse_args()


def main():
    # fix the random seed
    np.random.seed(args.seed)
    nowcast(args)


def nowcast(args):
    # nowcast
    print('Loading data...')
    dataset = SampleDataset(args.data_path, args.sample_index, args.input_steps, args.forecast_steps, args.lon_range, args.lat_range)
    tensor, timestamp = dataset[0]
    tensor = tensor.squeeze()
    input_, truth = tensor[:args.input_steps], tensor[args.input_steps:]
    input_[input_ < 0] = 0
    truth[truth < 0] = 0

    print('Nowcasting...')
    velocity = pysteps.motion.get_method('DARTS')(input_.numpy())
    pred = pysteps.nowcasts.get_method('sprog')(input_.numpy(), velocity, timesteps=args.forecast_steps, R_thr=0)
    pred[np.isnan(pred)] = 0
    pred = torch.from_numpy(pred)

    # visualization
    print('Visualizing...')
    input_, pred, truth = input_.unsqueeze(1).unsqueeze(2), pred.unsqueeze(1).unsqueeze(2), \
                          truth.unsqueeze(1).unsqueeze(2)
    timestamp = timestamp.unsqueeze(1)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    visualizer.plot_map(input_, pred, truth, timestamp, args.output_path, stage='sample')
    
    # evaluation
    print('Evaluating...')
    pred_rev, truth_rev = pred, truth
    pred = scaler.minmax_norm(pred, args.vmax, args.vmin)
    truth = scaler.minmax_norm(truth, args.vmax, args.vmin)
    
    metrics = {}
    metrics['Time'] = np.linspace(6, 60, 10)
    for threshold in args.thresholds:
        pod, far, csi, hss = evaluation.evaluate_forecast(pred_rev, truth_rev, threshold)
        metrics['POD-{}dBZ'.format(str(threshold))] = pod
        metrics['FAR-{}dBZ'.format(str(threshold))] = far
        metrics['CSI-{}dBZ'.format(str(threshold))] = csi
        metrics['HSS-{}dBZ'.format(str(threshold))] = hss

    metrics['CC'] = evaluation.evaluate_cc(pred_rev, truth_rev)
    metrics['ME'] = evaluation.evaluate_me(pred_rev, truth_rev)
    metrics['MAE'] = evaluation.evaluate_mae(pred_rev, truth_rev)
    metrics['RMSE'] = evaluation.evaluate_rmse(pred_rev, truth_rev)
    metrics['SSIM'] = evaluation.evaluate_ssim(pred, truth)
    metrics['PSNR'] = evaluation.evaluate_psnr(pred, truth)
    metrics['CVR'] = evaluation.evaluate_cvr(pred_rev, truth_rev)
    metrics['SSDR'] = evaluation.evaluate_ssdr(pred_rev, truth_rev)
    
    df = pd.DataFrame(data=metrics)
    df.to_csv(os.path.join(args.output_path, 'sample_metrics.csv'), float_format='%.8f', index=False)

    print('\nBaseline Done.')


if __name__ == '__main__':
    main()
