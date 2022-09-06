import os
import argparse
import warnings
import torch

from model import *
from utils.trainer import *
import utils.dataloader as dataloader


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

# input and output settings
parser.add_argument('--data-path', type=str, default='/data/gaf/SBandCRUnzip')
parser.add_argument('--output-path', type=str, default='results/AttnUNet')

# data loading settings
parser.add_argument('--train-ratio', type=float, default=0.7)
parser.add_argument('--valid-ratio', type=float, default=0.1)
parser.add_argument('--sample-index', type=int, default=0)

# training settings
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--early-stopping', action='store_true')

parser.add_argument('--hole_min_w', type=int, default=48)
parser.add_argument('--hole_max_w', type=int, default=96)
parser.add_argument('--hole_min_h', type=int, default=48)
parser.add_argument('--hole_max_h', type=int, default=96)
parser.add_argument('--ld_input_size', type=int, default=96)
parser.add_argument('--cn_input_size', type=int, default=256)

parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--max-iterations', type=int, default=100000)
parser.add_argument('--start-iterations', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--display-interval', type=int, default=1)
parser.add_argument('--seed', type=int, default=2022)

# nowcasting settings
parser.add_argument('--lon-range', type=int, nargs='+', default=[271, 527])
parser.add_argument('--lat-range', type=int, nargs='+', default=[335, 591])
parser.add_argument('--vmax', type=float, default=70.0)
parser.add_argument('--vmin', type=float, default=0.0)


def main(args):
    # fix the random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # Set device
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Set the model
    generator = CompletionNetwork()
    discriminator = ContextDiscriminator(local_input_shape=(1, args.ld_input_size, args.ld_input_size),
                                         global_input_shape=(1, args.cn_input_size, args.cn_input_size),)
 
    # Load data
    if args.train or args.test:
        train_loader, val_loader, test_loader = dataloader.load_data(args.data_path, 
            args.batch_size, args.num_workers, args.train_ratio, args.valid_ratio, 
            args.lon_range, args.lat_range)
    if args.predict:
        sample_loader = dataloader.load_sample(args.data_path, args.sample_index, 
            args.lon_range, args.lat_range)

    # Nowcasting
    print('\nStart tasks...')

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    trainer = Trainer(args)
    if args.train or args.test:
        trainer.fit(generator, discriminator, train_loader, val_loader, test_loader)
    if args.predict:
        trainer.predict(generator, sample_loader)
        
    print('\nAll tasks have finished.')
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
