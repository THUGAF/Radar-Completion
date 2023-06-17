import os
import sys
sys.path.append(os.getcwd())
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.interpolate import RegularGridInterpolator
import utils.dataloader as dataloader
import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.maskutils as maskutils


parser = argparse.ArgumentParser()

# input and output settings
parser.add_argument('--data-path', type=str)
parser.add_argument('--output-path', type=str, default='results')
parser.add_argument('--elevation-id', type=int, nargs='+', default=[1])
parser.add_argument('--azimuthal-range', type=int, nargs='+', default=[0, 360])
parser.add_argument('--radial-range', type=int, nargs='+', default=[0, 80])

# data loading settings
parser.add_argument('--train-ratio', type=float, default=0.7)
parser.add_argument('--valid-ratio', type=float, default=0.1)

# mask settings
parser.add_argument('--azimuth-blockage-range', type=int, nargs='+', default=[10, 40])
parser.add_argument('--case-indices', type=int, nargs='+', default=[0])
parser.add_argument('--case-anchor', type=int, nargs='+', default=[0])
parser.add_argument('--case-blockage-len', type=int, nargs='+', default=[40])

# training settings
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--display-interval', type=int, default=1)
parser.add_argument('--random-seed', type=int, default=2023)

args = parser.parse_args()


def main(args):
    print('### Initialize settings ###')

    # Fix the random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Make dir
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Test, and predict
    print('\n### Start tasks ###')
    if args.test:
        test_loader = dataloader.load_data(
            args.data_path, args.batch_size, args.num_workers, args.train_ratio, args.valid_ratio,
            args.elevation_id, args.azimuthal_range, args.radial_range)[2]
        test(test_loader)
    if args.predict:
        case_loader = dataloader.load_case(
            args.data_path, args.case_indices,
            args.elevation_id, args.azimuthal_range, args.radial_range)
        predict(case_loader)
    
    print('\n### All tasks complete ###')


def direct_filling(ref: torch.Tensor, azimuth_start_point: int, anchor: int, 
                   blockage_len: int, used_elevation_order: int = -1) -> torch.Tensor:
    new_ref = torch.clone(ref)
    new_ref[:, 0, anchor - azimuth_start_point: 
            anchor + blockage_len - azimuth_start_point] = \
    new_ref[:, used_elevation_order, anchor - azimuth_start_point: 
            anchor + blockage_len - azimuth_start_point]
    new_ref = new_ref[:, :1]
    return new_ref


def biliear_interp(masked_ref: torch.Tensor, azimuth_start_point: int, anchor: int,
                   blockage_len: int) -> torch.Tensor:
    batch_size, _, azimuthal_len, radial_len = masked_ref.size()
    unmasked_ref = torch.cat([masked_ref[:, :, : anchor - azimuth_start_point], 
                              masked_ref[:, :, anchor + blockage_len - azimuth_start_point:]], 
                              dim=2)
    b, c, r = np.arange(batch_size), np.arange(1), np.arange(radial_len)
    a = np.concatenate([np.arange(anchor - azimuth_start_point),
                        np.arange(anchor + blockage_len - azimuth_start_point, azimuthal_len)])
    
    interp = RegularGridInterpolator((b, c, a, r), unmasked_ref[:, :1].numpy())
    full_a = np.arange(azimuthal_len)
    b_grid, c_grid, full_a_grid, r_grid = np.meshgrid(b, c, full_a, r, indexing='ij')
    interp_ref = torch.from_numpy(interp((b_grid, c_grid, full_a_grid, r_grid)))
    return interp_ref


@torch.no_grad()
def test(test_loader):
    metrics = {}
    metrics['MAE'] = 0
    metrics['RMSE'] = 0
    metrics['MBE'] = 0

    # Timer
    test_timer = time.time()
    test_batch_timer = time.time()
    
    print('\n[Test]')
    for i, (t, elev, ref) in enumerate(test_loader):
        # Forward
        masked_ref, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
            ref, args.azimuth_blockage_range, args.random_seed + i)
        output = biliear_interp(masked_ref, args.azimuthal_range[0], anchor, blockage_len)
        output = torch.clip(output, min=0, max=70)
        output = torch.nan_to_num(output)

        # Print time
        if (i + 1) % args.display_interval == 0:
            print('Batch: [{}][{}]\tTime: {:.4f}'.format(
                i + 1, len(test_loader), time.time() - test_batch_timer))
            test_batch_timer = time.time()

        # Evaluation
        truth = torch.clip(ref[:, :1], min=0, max=70)
        total_mae = evaluation.evaluate_mae(output, truth, mask[:, :1])
        total_rmse = evaluation.evaluate_rmse(output, truth, mask[:, :1])
        total_mbe = evaluation.evaluate_mbe(output, truth, mask[:, :1])
        thresholds, maes = evaluation.evaluate_mae_multi_thresholds(output, truth, mask[:, :1])
        thresholds, rmses = evaluation.evaluate_rmse_multi_thresholds(output, truth, mask[:, :1])
        thresholds, mbes = evaluation.evaluate_mbe_multi_thresholds(output, truth, mask[:, :1])
        metrics['MAE'] += np.append(maes, total_mae)
        metrics['RMSE'] += np.append(rmses, total_rmse)
        metrics['MBE'] += np.append(mbes, total_mbe)

    # Print test time
    print('Time: {:.4f}'.format(time.time() - test_timer))

    # Save metrics
    for key in metrics.keys():
        metrics[key] /= len(test_loader)
    index = [str(t) for t in thresholds] + ['total']
    df = pd.DataFrame(data=metrics, index=index)
    df.to_csv(os.path.join(args.output_path, 'test_metrics.csv'), float_format='%.4f')
    print('Test metrics saved')


@torch.no_grad()
def predict(case_loader: DataLoader):
    print('\n[Predict]')
    for i, (t, elev, ref) in enumerate(case_loader):
        print('\nCase {} at {}'.format(i, t))
        metrics = {}

        # Forward
        masked_ref, mask, anchor, blockage_len = maskutils.gen_fixed_blockage_mask(
            ref, args.azimuthal_range[0], args.case_anchor[i], args.case_blockage_len[i])
        output = biliear_interp(masked_ref, args.azimuthal_range[0], anchor, blockage_len)
        output = torch.clip(output, min=0, max=70)
        output = torch.nan_to_num(output)

        # Evaluation
        truth = torch.clip(ref[:, :1], min=0, max=70)
        total_mae = evaluation.evaluate_mae(output, truth, mask[:, :1])
        total_rmse = evaluation.evaluate_rmse(output, truth, mask[:, :1])
        total_mbe = evaluation.evaluate_mbe(output, truth, mask[:, :1])
        thresholds, maes = evaluation.evaluate_mae_multi_thresholds(output, truth, mask[:, :1])
        thresholds, rmses = evaluation.evaluate_rmse_multi_thresholds(output, truth, mask[:, :1])
        thresholds, mbes = evaluation.evaluate_mbe_multi_thresholds(output, truth, mask[:, :1])
        metrics['MAE'] = np.append(maes, total_mae)
        metrics['RMSE'] = np.append(rmses, total_rmse)
        metrics['MBE'] = np.append(mbes, total_mbe)

        index = [str(t) for t in thresholds] + ['total']
        df = pd.DataFrame(data=metrics, index=index)
        df.to_csv(os.path.join(args.output_path, 'case_{}_metrics.csv'.format(i)), float_format='%.4f')
        print('Case {} metrics saved'.format(i))

        # Save tensors
        tensors = torch.cat([output, ref], dim=1)
        visualizer.save_tensor(tensors, args.output_path, 'case_{}'.format(i))
        print('Tensors saved')

        # Plot tensors
        visualizer.plot_ppi(tensors, t, args.azimuthal_range[0], args.radial_range[0],
                            anchor, blockage_len, args.output_path, 'case_{}'.format(i))
        visualizer.plot_psd(tensors, args.radial_range[0], anchor, blockage_len,
                            args.output_path, 'case_{}'.format(i))
        print('Figures saved')

    print('\nPrediction complete')


if __name__ == '__main__':
    main(args)
