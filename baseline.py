import os
import argparse
import warnings
import torch

from utils.baselinetester import *
import utils.dataloader as dataloader


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# input and output settings
parser.add_argument('--data-path', type=str, default='/data/gaf/SBandBasicUnzip')
parser.add_argument('--output-path', type=str, default='results')
parser.add_argument('--elevation-id', type=int, nargs='+', default=[1, 2])
parser.add_argument('--azimuth-range', type=int, nargs='+', default=[0, 360])
parser.add_argument('--radial-range', type=int, nargs='+', default=[0, 80])

# data loading settings
parser.add_argument('--baseline-method', type=str, default='direct')
parser.add_argument('--train-ratio', type=float, default=0.7)
parser.add_argument('--valid-ratio', type=float, default=0.1)
parser.add_argument('--vmax', type=float, default=70.0)
parser.add_argument('--vmin', type=float, default=-10.0)

# mask settings
parser.add_argument('--azimuth-blockage-range', type=int, nargs='+', default=[10, 20])
parser.add_argument('--sample-index', type=int, nargs='+', default=[0])
parser.add_argument('--sample-anchor', type=int, default=0)
parser.add_argument('--sample-blockage-len', type=int, default=40)

# training settings
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--display-interval', type=int, default=1)
parser.add_argument('--random-seed', type=int, default=2023)


def main(args):
    # fix the random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.autograd.set_detect_anomaly(True)

    # Load data
    if args.test:
        test_loader = dataloader.load_data(
            args.data_path, args.batch_size, args.num_workers, args.train_ratio, args.valid_ratio,
            args.elevation_id, args.azimuth_range, args.radial_range)[2]
    if args.predict:
        sample_loader = dataloader.load_sample(
            args.data_path, args.sample_index,
            args.elevation_id, args.azimuth_range, args.radial_range)
    
    # Nowcasting
    print('\nStart tasks...')

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    tester = BaselineTester(args)
    if args.test:
        tester.test(test_loader)
    if args.predict:
        tester.predict(sample_loader)
    
    print('\nAll tasks have finished.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
