import os
import argparse
import warnings
import torch

from model import *
from utils.earlystopping import *
import utils.dataloader as dataloader


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# input and output settings
parser.add_argument('--data-path', type=str)
parser.add_argument('--output-path', type=str, default='results')
parser.add_argument('--elevation-id', type=int, nargs='+', default=[1, 2])
parser.add_argument('--azimuth-range', type=int, nargs='+', default=[0, 360])
parser.add_argument('--radial-range', type=int, nargs='+', default=[0, 80])

# data loading settings
parser.add_argument('--train-ratio', type=float, default=0.64)
parser.add_argument('--valid-ratio', type=float, default=0.16)
parser.add_argument('--vmax', type=float, default=70.0)
parser.add_argument('--vmin', type=float, default=-10.0)

# mask settings
parser.add_argument('--azimuth-blockage-range', type=int, nargs='+', default=[10, 40])
parser.add_argument('--sample-index', type=int, nargs='+', default=[0])
parser.add_argument('--sample-anchor', type=int, nargs='+', default=[0])
parser.add_argument('--sample-blockage-len', type=int, nargs='+', default=[40])

# model settings
parser.add_argument('--model', type=str,
                    choices=['GLCIC_GAN', 'UNetpp_GAN', 'DilatedUNet'])

# training settings
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--max-iterations', type=int, default=100000)
parser.add_argument('--start-iterations', type=int, default=0)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--weight-recon', type=float, default=100)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--display-interval', type=int, default=1)
parser.add_argument('--random-seed', type=int, default=2023)

args = parser.parse_args()


def main(args):
    # fix the random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # Set device
    torch.set_num_threads(args.num_threads)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Set the model
    model = eval(args.model)(args)
    model = model.to(args.device)

    # Load data
    if args.train or args.test:
        train_loader, val_loader, test_loader = dataloader.load_data(
            args.data_path, args.batch_size, args.num_workers, args.train_ratio, args.valid_ratio, 
            args.elevation_id, args.azimuth_range, args.radial_range)
    if args.predict:
        sample_loader = dataloader.load_sample(
            args.data_path, args.sample_index, 
            args.elevation_id, args.azimuth_range, args.radial_range)

    # Nowcasting
    print('\nStart tasks...')

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if args.model == 'UNetpp_GAN':
        trainer = UNetpp_GAN_Trainer(args)
    elif args.model == 'GLCIC_GAN':
        trainer = GLCIC_GAN_Trainer(args)
    elif args.model == 'DilatedUNet':
        trainer = DilatedUNet_Trainer(args)

    if args.train or args.test:
        trainer.fit(model, train_loader, val_loader, test_loader)
    if args.predict:
        trainer.predict(model, sample_loader)
        
    print('\nAll tasks have finished.')
    

if __name__ == '__main__':
    main(args)
