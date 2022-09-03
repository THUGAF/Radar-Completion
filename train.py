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
parser.add_argument('--input-steps', type=int, default=10)
parser.add_argument('--forecast-steps', type=int, default=10)

# data loading settings
parser.add_argument('--train-ratio', type=float, default=0.7)
parser.add_argument('--valid-ratio', type=float, default=0.1)
parser.add_argument('--sample-index', type=int, default=0)

# model settings
parser.add_argument('--model', type=str, default='AttnUNet')
parser.add_argument('--add-gan', action='store_true')
parser.add_argument('--rolling', action='store_true')

# training settings
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--ensemble-members', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--max-iterations', type=int, default=100000)
parser.add_argument('--start-iterations', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=1e-2)
parser.add_argument('--var-reg', type=float, default=0)
parser.add_argument('--gan-reg', type=float, default=0.1)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--display-interval', type=int, default=1)
parser.add_argument('--seed', type=int, default=2022)

# nowcasting settings
parser.add_argument('--resolution', type=float, default=6.0, help='Time resolution (min)')
parser.add_argument('--lon-range', type=int, nargs='+', default=[271, 527])
parser.add_argument('--lat-range', type=int, nargs='+', default=[335, 591])
parser.add_argument('--vmax', type=float, default=70.0)
parser.add_argument('--vmin', type=float, default=0.0)

# evaluation settings
parser.add_argument('--thresholds', type=int, nargs='+', default=[10, 15, 20, 25, 30, 35, 40])

args = parser.parse_args()


def main():
    # Display global settings
    print('Temporal resolution: {} min'.format(args.resolution))
    print('Spatial resolution: 1.0 km')
    print('Input steps: {}'.format(str(args.input_steps)))
    print('Forecast steps: {}'.format(str(args.forecast_steps)))
    print('Input time range: {} min'.format(str(args.input_steps * args.resolution)))
    print('Forecast time range: {} min'.format(str(args.forecast_steps * args.resolution)))

    # fix the random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # Set device
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Set the model
    if args.rolling:
        if args.model == 'AttnUNet':
            model = AttnUNet(args.input_steps, 1)
        elif args.model == 'EncoderForecaster':
            model = EncoderForecaster(1, 1, 1, [64, 64, 64, 64])
        elif args.model == 'SmaAt_UNet':
            model = SmaAt_UNet(args.input_steps, 1)
        if args.add_gan:
            model = AttnUNet(args.input_steps, 1, add_noise=True)
            model = GAN(model, args)
    else:
        if args.model == 'AttnUNet':
            model = AttnUNet(args.input_steps, args.forecast_steps)
        elif args.model == 'EncoderForecaster': 
            model = EncoderForecaster(args.forecast_steps, 1, 1, [64, 64, 64, 64])
        elif args.model == 'SmaAt_UNet': 
            model = SmaAt_UNet(args.input_steps, args.forecast_steps)
        if args.add_gan:
            model = AttnUNet(args.input_steps, args.forecast_steps, add_noise=True)
            model = GAN(model, args)
 
    # Load data
    if args.train or args.test:
        train_loader, val_loader, test_loader = dataloader.load_data(args.data_path, 
            args.input_steps, args.forecast_steps, args.batch_size, args.num_workers, 
            args.train_ratio, args.valid_ratio, args.lon_range, args.lat_range)
    if args.predict:
        sample_loader = dataloader.load_sample(args.data_path, args.sample_index, args.input_steps, 
            args.forecast_steps, args.lon_range, args.lat_range)

    # Nowcasting
    print('\nStart tasks...')
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    # Set trainer
    if args.rolling:
        if args.add_gan:
            trainer = RollingGANTrainer(args)
        else:
            trainer = RollingTrainer(args)
    else:
        if args.add_gan:
            trainer = GANTrainer(args)
        else:
            trainer = Trainer(args)
    if args.train or args.test:
        trainer.fit(model, train_loader, val_loader, test_loader)
    if args.predict:
        trainer.predict(model, sample_loader)
        
    print('\nAll tasks have finished.')
    

if __name__ == '__main__':
    main()
