import os
import sys
sys.path.append(os.getcwd())
import time
import shutil
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.transform as transform
import utils.dataloader as dataloader
import utils.maskutils as maskutils
import models


parser = argparse.ArgumentParser()

# input and output settings
parser.add_argument('--data-path', type=str)
parser.add_argument('--output-path', type=str, default='results')
parser.add_argument('--elevation-id', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--azimuthal-range', type=int, nargs='+', default=[0, 360])
parser.add_argument('--radial-range', type=int, nargs='+', default=[0, 80])
parser.add_argument('--padding-width', type=int, default=20)

# data loading settings
parser.add_argument('--train-ratio', type=float, default=0.7)
parser.add_argument('--valid-ratio', type=float, default=0.1)

# mask settings
parser.add_argument('--azimuth-blockage-range', type=int, nargs='+', default=[10, 40])
parser.add_argument('--case-indices', type=int, nargs='+', default=[0])
parser.add_argument('--case-anchor', type=int, nargs='+', default=[0])
parser.add_argument('--case-blockage-len', type=int, nargs='+', default=[40])

# training settings
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--augment-ratio', type=int, default=1)
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
    print('### Initialize settings ###')

    # Fix the random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set device
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda:0'
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    else:
        args.device = 'cpu'

    # Set model and optimizer
    model = models.UNetpp_GAN(args).to(args.device)
    count_params(model)
    optimizer_g = optim.Adam(model.generator.parameters(), args.learning_rate,
                             betas=(args.beta1, args.beta2),
                             weight_decay=args.weight_decay)
    optimizer_d = optim.Adam(model.discriminator.parameters(), args.learning_rate * 2,
                             betas=(args.beta1, args.beta2),
                             weight_decay=args.weight_decay)

    # Make dir
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Load data
    if args.train or args.test:
        train_loader, val_loader, test_loader = dataloader.load_data(
            args.data_path, args.batch_size, args.num_workers, args.train_ratio, args.valid_ratio,
            args.elevation_id, args.azimuthal_range, args.radial_range, args.augment_ratio)

    # Train, test, and predict
    print('\n### Start tasks ###')
    if args.train:
        train(model, optimizer_g, optimizer_d, train_loader, val_loader)
    if args.test:
        test(model, test_loader)
    if args.predict:
        case_loader = dataloader.load_case(args.data_path, args.case_indices, args.elevation_id,
                                           args.azimuthal_range, args.radial_range)
        predict(model, case_loader)

    print('\n### All tasks complete ###')


def count_params(model: nn.Module):
    G_params = filter(lambda p: p.requires_grad, model.generator.parameters())
    D_params = filter(lambda p: p.requires_grad,
                      model.discriminator.parameters())
    num_G_params = sum([p.numel() for p in G_params])
    num_D_params = sum([p.numel() for p in D_params])
    print('\nModel name: {}'.format(type(model).__name__))
    print('G params: {}'.format(num_G_params))
    print('D params: {}'.format(num_D_params))
    print('Total params: {}'.format(num_G_params + num_D_params))


def save_checkpoint(filename: str, current_iteration: int, train_loss_g: list, train_loss_d: list,
                    val_loss_g: list, val_loss_d: list, model: nn.Module,
                    optimizer_g: optim.Optimizer, optimizer_d: optim.Optimizer):
    states = {
        'iteration': current_iteration,
        'train_loss_g': train_loss_g,
        'train_loss_d': train_loss_d,
        'val_loss_g': val_loss_g,
        'val_loss_d': val_loss_d,
        'generator': model.generator.state_dict(),
        'discriminator': model.discriminator.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict()
    }
    torch.save(states, filename)


def load_checkpoint(filename: str, device: str):
    states = torch.load(filename, map_location=device)
    return states


def early_stopping(val_loss: list, patience: int = 10):
    early_stopping_flag = False
    counter = 0
    current_epoch = len(val_loss)
    if current_epoch == 1:
        min_val_loss = np.inf
    else:
        min_val_loss = min(val_loss[:-1])
    if min_val_loss > val_loss[-1]:
        print(
            'Validation loss decreased: {:.4f} --> {:.4f}'.format(min_val_loss, val_loss[-1]))
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        bestparams_path = os.path.join(args.output_path, 'bestparams.pth')
        shutil.copyfile(checkpoint_path, bestparams_path)
    else:
        min_val_loss_epoch = val_loss.index(min(val_loss))
        if current_epoch > min_val_loss_epoch:
            counter = current_epoch - min_val_loss_epoch
            print('EarlyStopping counter: {} out of {}'.format(counter, patience))
            if counter == patience:
                early_stopping_flag = True
    return early_stopping_flag


def weighted_l1_loss(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    points = torch.tensor([10.0, 20.0, 30.0, 40.0])
    points = transform.minmax_norm(points)
    weight = (truth < points[0]) * 1 \
        + (torch.logical_and(truth >= points[0], truth < points[1])) * 2 \
        + (torch.logical_and(truth >= points[1], truth < points[2])) * 5 \
        + (torch.logical_and(truth >= points[2], truth < points[3])) * 10 \
        + (truth >= points[3]) * 30
    return torch.mean(weight * torch.abs(pred - truth))


def d_loss(fake_score: torch.Tensor, real_score: torch.Tensor, loss_func=nn.BCELoss()) -> torch.Tensor:
    label = torch.ones_like(fake_score).type_as(fake_score)
    loss_pred = loss_func(fake_score, label * 0.0)
    loss_truth = loss_func(real_score, label * 1.0)
    return (loss_pred + loss_truth) / 2


def g_loss(fake_score: torch.Tensor, loss_func=nn.BCELoss()) -> torch.Tensor:
    label = torch.ones_like(fake_score).type_as(fake_score)
    loss = loss_func(fake_score, label * 1.0)
    return loss


def pad_azimuth(ref: torch.Tensor, width: int = 20) -> torch.Tensor:
    start_pad = ref[:, :, -width:]
    end_pad = ref[:, :, :width]
    ref = torch.cat([start_pad, ref, end_pad], dim=2)
    return ref


def unpad_azimuth(ref: torch.Tensor, width: int = 20) -> torch.Tensor:
    return ref[:, :, width: -width]


def train(model: nn.Module, optimizer_g: optim.Optimizer, optimizer_d: optim.Optimizer,
          train_loader: DataLoader, val_loader: DataLoader):
    # Pretrain: Load model and optimizer
    if args.pretrain:
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        states = load_checkpoint(checkpoint_path, args.device)
        current_iteration = states['iteration']
        train_loss_g = states['train_loss_g']
        train_loss_d = states['train_loss_d']
        val_loss_g = states['val_loss_g']
        val_loss_d = states['val_loss_d']
        model.generator.load_state_dict(states['generator'])
        model.discriminator.load_state_dict(states['discriminator'])
        optimizer_g.load_state_dict(states['optimizer_g'])
        optimizer_d.load_state_dict(states['optimizer_d'])
        start_epoch = int(np.floor(current_iteration / len(train_loader)))
    else:
        current_iteration = 0
        train_loss_g = []
        train_loss_d = []
        val_loss_g = []
        val_loss_d = []
        start_epoch = 0
    
    # Train and validation
    total_epochs = int(np.ceil((args.max_iterations - current_iteration) / len(train_loader)))
    print('\nMax iterations:', args.max_iterations)
    print('Total epochs:', total_epochs)

    for epoch in range(start_epoch, total_epochs):
        train_loss_g_epoch = 0
        train_loss_d_epoch = 0
        val_loss_g_epoch = 0
        val_loss_d_epoch = 0

        # Train
        print('\n[Train]')
        print('Epoch: [{}][{}]'.format(epoch + 1, total_epochs))
        model.train()

        # Timers
        train_epoch_timer = time.time()
        train_batch_timer = time.time()

        for i, (t, elev, ref) in enumerate(train_loader):
            # Check max iterations
            current_iteration += 1
            if current_iteration > args.max_iterations:
                print('Max iterations reached. Exit!')
                break       
            
            # load data
            ref = ref.to(args.device)
            ref = pad_azimuth(ref, args.padding_width)
            ref_norm = transform.minmax_norm(ref)
            masked_ref_norm, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
                ref_norm, args.azimuth_blockage_range, args.random_seed + i)
            
            # discriminator forward
            output_g_norm = model(torch.cat([masked_ref_norm, mask], dim=1))
            output_g_norm = masked_ref_norm[:, :1] + output_g_norm * (1 - mask[:, :1])
            real_input_g = torch.cat([ref_norm[:, :1], mask[:, :1]], dim=1)
            fake_input_g = torch.cat([output_g_norm.detach(), mask[:, :1]], dim=1)
            real_score = model.discriminator(real_input_g)
            fake_score = model.discriminator(fake_input_g)
            loss_d = d_loss(fake_score, real_score)
            
            # discriminator backward
            optimizer_d.zero_grad()
            loss_d.backward()
            clip_grad_norm_(model.discriminator.parameters(), 1e-4)
            optimizer_d.step()

            # generator backward
            fake_input_g = torch.cat([output_g_norm, mask[:, :1]], dim=1)
            fake_score = model.discriminator(fake_input_g)
            loss_g = g_loss(fake_score) + args.weight_recon * weighted_l1_loss(
                output_g_norm, ref_norm[:, :1])
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            
            # Record and print loss
            train_loss_g_epoch += loss_g.item()
            train_loss_d_epoch += loss_d.item()
            if (i + 1) % args.display_interval == 0:
                print('Epoch: [{}][{}]\tBatch: [{}][{}]\tLoss G: {:.4f}\tLoss D: {:.4f}\tTime: {:.4f}'.format(
                    epoch + 1, total_epochs, i + 1, len(train_loader),
                    loss_g.item(), loss_d.item(), time.time() - train_batch_timer))
                train_batch_timer = time.time()
        
        # Save train loss
        train_loss_g_epoch = train_loss_g_epoch / len(train_loader)
        train_loss_d_epoch = train_loss_g_epoch / len(train_loader)
        print('Epoch: [{}][{}]\tLoss G: {:.4f}\tLoss D: {:.4f}\tTime: {:.4f}'.format(
            epoch + 1, total_epochs, train_loss_g_epoch, train_loss_d_epoch, time.time() - train_epoch_timer))
        train_epoch_timer = time.time()
        train_loss_g.append(train_loss_g_epoch)
        train_loss_d.append(train_loss_d_epoch)
        np.savetxt(os.path.join(args.output_path, 'train_loss_g.txt'), train_loss_g)
        np.savetxt(os.path.join(args.output_path, 'train_loss_d.txt'), train_loss_d)
        
        # Validate
        print('\n[Validate]')
        print('Epoch: [{}][{}]'.format(epoch + 1, total_epochs))
        model.eval()

        # Timers
        val_epoch_timer = time.time()
        val_batch_timer = time.time()

        with torch.no_grad():
            for i, (t, elev, ref) in enumerate(val_loader):
                # load data
                ref = ref.to(args.device)
                ref = pad_azimuth(ref, args.padding_width)
                ref_norm = transform.minmax_norm(ref)
                masked_ref_norm, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
                    ref_norm, args.azimuth_blockage_range, args.random_seed + i)
                
                # discriminator forward
                output_g_norm = model(torch.cat([masked_ref_norm, mask], dim=1))
                output_g_norm = masked_ref_norm[:, :1] + output_g_norm * (1 - mask[:, :1])
                real_input_g = torch.cat([ref_norm[:, :1], mask[:, :1]], dim=1)
                fake_input_g = torch.cat([output_g_norm.detach(), mask[:, :1]], dim=1)
                real_score = model.discriminator(real_input_g)
                fake_score = model.discriminator(fake_input_g)
                loss_d = d_loss(fake_score, real_score)

                # generator forward
                loss_g = g_loss(fake_score) + args.weight_recon * weighted_l1_loss(
                    output_g_norm, ref_norm[:, :1])
                
                # Record and print loss
                val_loss_g_epoch += loss_g.item()
                val_loss_d_epoch += loss_d.item()
                if (i + 1) % args.display_interval == 0:
                    print('Epoch: [{}][{}]\tBatch: [{}][{}]\tLoss G: {:.4f}\tLoss D: {:.4f}\tTime: {:.4f}'.format(
                        epoch + 1, total_epochs, i + 1, len(val_loader),
                        loss_g.item(), loss_d.item(), time.time() - val_batch_timer))
                    val_batch_timer = time.time()

        # Save val loss
        val_loss_g_epoch = val_loss_g_epoch / len(val_loader)
        val_loss_d_epoch = val_loss_d_epoch / len(val_loader)
        print('Epoch: [{}][{}]\tLoss G: {:.4f}\tLoss D: {:.4f}\tTime: {:.4f}'.format(
            epoch + 1, total_epochs, val_loss_g_epoch, val_loss_d_epoch,
            time.time() - val_epoch_timer))
        val_epoch_timer = time.time()
        val_loss_g.append(val_loss_g_epoch)
        val_loss_d.append(val_loss_d_epoch)
        np.savetxt(os.path.join(args.output_path, 'val_loss_g.txt'), val_loss_g)
        np.savetxt(os.path.join(args.output_path, 'val_loss_d.txt'), val_loss_d)

        # Plot loss
        visualizer.plot_loss(train_loss_g, val_loss_g, args.output_path, 'loss_g.jpg')
        visualizer.plot_loss(train_loss_d, val_loss_d, args.output_path, 'loss_d.jpg')
        print('Loss figure saved')

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        save_checkpoint(checkpoint_path, current_iteration, train_loss_g, train_loss_d,
                        val_loss_g, val_loss_d, model, optimizer_g, optimizer_d)
        if args.early_stopping:
            early_stopping_flag = early_stopping(val_loss_g)
            if early_stopping_flag:
                print('Early stopped')
                break


@torch.no_grad()
def test(model: nn.Module, test_loader: DataLoader):
    # Init metric dict
    metrics = {}
    metrics['MAE'] = 0
    metrics['RMSE'] = 0
    metrics['MBE'] = 0
    
    # Test
    print('\n[Test]')
    bestparams_path = os.path.join(args.output_path, 'bestparams.pth')
    states = load_checkpoint(bestparams_path, args.device)
    model.generator.load_state_dict(states['generator'])
    model.discriminator.load_state_dict(states['discriminator'])
    model.eval()

    # Timer
    test_timer = time.time()
    test_batch_timer = time.time()

    for i, (t, elev, ref) in enumerate(test_loader):
        # Forward propagation
        ref = ref.to(args.device)
        ref = pad_azimuth(ref, args.padding_width)
        ref_norm = transform.minmax_norm(ref)
        masked_ref_norm, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
            ref_norm, args.azimuth_blockage_range, args.random_seed + i)
        output_norm = model(torch.cat([masked_ref_norm, mask], dim=1))
        output_norm = masked_ref_norm[:, :1] + output_norm * (1 - mask[:, :1])

        # Print time
        if (i + 1) % args.display_interval == 0:
            print('Batch: [{}][{}]\tTime: {:.4f}'.format(
                i + 1, len(test_loader), time.time() - test_batch_timer))
            test_batch_timer = time.time()

        # Back scaling
        ref = transform.reverse_minmax_norm(ref_norm)
        output = transform.reverse_minmax_norm(output_norm)

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
def predict(model: nn.Module, case_loader: DataLoader):   
    # Predict
    print('\n[Predict]')
    bestparams_path = os.path.join(args.output_path, 'bestparams.pth')
    states = load_checkpoint(bestparams_path, args.device)
    model.generator.load_state_dict(states['generator'])
    model.eval()
    for i, (t, elev, ref) in enumerate(case_loader):
        t = datetime.datetime.strptime(
            str(t.item()), '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
        print('\nCase {} at {}'.format(i, t))
        metrics = {}

        # Forward propagation
        ref = ref.to(args.device)
        ref = pad_azimuth(ref, args.padding_width)
        ref_norm = transform.minmax_norm(ref)
        masked_ref_norm, mask, anchor, blockage_len = maskutils.gen_fixed_blockage_mask(
            ref_norm, args.azimuthal_range[0], args.case_anchor[i] + args.padding_width, args.case_blockage_len[i])
        output_norm = model(torch.cat([masked_ref_norm, mask], dim=1))
        output_norm = masked_ref_norm[:, :1] + output_norm * (1 - mask[:, :1])

        # Back scaling
        ref = transform.reverse_minmax_norm(ref_norm)
        output = transform.reverse_minmax_norm(output_norm)

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
        tensors = unpad_azimuth(torch.cat([output, ref], dim=1), args.padding_width)
        visualizer.save_tensor(tensors, args.output_path, 'case_{}'.format(i))
        print('Tensors saved')
        
        # Plot tensors
        visualizer.plot_ppi(tensors, t, args.azimuthal_range[0], args.radial_range[0],
                            anchor - args.padding_width, blockage_len, args.output_path, 'case_{}'.format(i))
        visualizer.plot_psd(tensors, args.radial_range[0], anchor - args.padding_width, blockage_len,
                            args.output_path, 'case_{}'.format(i))
        print('Figures saved')

    print('\nPrediction complete')


if __name__ == '__main__':
    main(args)
