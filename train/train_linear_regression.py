import os
import sys
sys.path.append(os.getcwd())
import time
import shutil
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.dataloader as dataloader
import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.maskutils as maskutils


parser = argparse.ArgumentParser()

# input and output settings
parser.add_argument('--data-path', type=str)
parser.add_argument('--output-path', type=str, default='results')
parser.add_argument('--elevation-id', type=int, nargs='+', default=[1, 2, 3])
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
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--max-iterations', type=int, default=100000)
parser.add_argument('--start-iterations', type=int, default=0)
parser.add_argument('--learning-rate', type=float, default=0.001)
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
    model = nn.Linear(in_features=len(args.elevation_id) - 1, out_features=1).to(args.device)
    count_params(model)
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    # Make dir
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Load data
    if args.train or args.test:
        train_loader, val_loader, test_loader = dataloader.load_data(
            args.data_path, args.batch_size, args.num_workers, args.train_ratio, args.valid_ratio,
            args.elevation_id, args.azimuthal_range, args.radial_range)
    
    # Train, test, and predict
    print('\n### Start tasks ###')
    if args.train:
        train(model, optimizer, train_loader, val_loader)
    if args.test:
        test(model, test_loader)
    if args.predict:
        case_loader = dataloader.load_case(
            args.data_path, args.case_indices,
            args.elevation_id, args.azimuthal_range, args.radial_range)
        predict(model, case_loader)
    
    print('\n### All tasks complete ###')
    

def count_params(model: nn.Module):
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([p.numel() for p in model_params])
    print('\nModel name: {}'.format(type(model).__name__))
    print('Total params: {}'.format(num_params))


def save_checkpoint(filename: str, current_iteration: int, train_loss: list, val_loss: list,
                    model: nn.Module, optimizer: optim.Optimizer):
    states = {
        'iteration': current_iteration,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
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
        print('Validation loss decreased: {:.4f} --> {:.4f}'.format(min_val_loss, val_loss[-1]))
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


def train(model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader):
    # Pretrain
    if args.pretrain:
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        states = load_checkpoint(checkpoint_path, args.device)
        current_iteration = states['iteration']
        train_loss = states['train_loss']
        val_loss = states['val_loss']
        model.load_state_dict(states['model'])
        optimizer.load_state_dict(states['optimizer'])
        start_epoch = int(np.floor(current_iteration / len(train_loader)))
    else:
        current_iteration = 0
        train_loss = []
        val_loss = []
        start_epoch = 0

    # Train and validation
    total_epochs = int(np.ceil((args.max_iterations - current_iteration) / len(train_loader)))
    print('\nMax iterations:', args.max_iterations)
    print('Total epochs:', total_epochs)

    for epoch in range(start_epoch, total_epochs):
        train_loss_epoch = 0
        val_loss_epoch = 0

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
        
            # Forward propagation
            ref = ref.to(args.device)
            input_, truth = ref[:, 1:], ref[:, :1]
            input_ = input_.permute(0, 2, 3, 1).contiguous()
            output = model(input_).permute(0, 3, 1, 2).contiguous()
            loss = nn.MSELoss()(output, truth)

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record and print loss
            train_loss_epoch += loss.item()
            if (i + 1) % args.display_interval == 0:
                print('Epoch: [{}][{}]\tBatch: [{}][{}]\tLoss: {:.4f}\tTime: {:.4f}'.format(
                    epoch + 1, total_epochs, i + 1, len(train_loader), loss.item(), 
                    time.time() - train_batch_timer))
                train_batch_timer = time.time()
        
        # Save train loss
        train_loss_epoch = train_loss_epoch / len(train_loader)
        print('Epoch: [{}][{}]\tLoss: {:.4f}\tTime: {:.4f}'.format(
            epoch + 1, total_epochs, train_loss_epoch, time.time() - train_epoch_timer))
        train_epoch_timer = time.time()
        train_loss.append(train_loss_epoch)
        np.savetxt(os.path.join(args.output_path, 'train_loss.txt'), train_loss)
        print('Train loss saved')

        # Validate
        print('\n[Validate]')
        print('Epoch: [{}][{}]'.format(epoch + 1, total_epochs))
        model.eval()

        # Timers
        val_epoch_timer = time.time()
        val_batch_timer = time.time()

        with torch.no_grad():
            for i, (t, elev, ref) in enumerate(val_loader):
                # Forward propagation
                ref = ref.to(args.device)
                input_, truth = ref[:, 1:], ref[:, :1]
                input_ = input_.permute(0, 2, 3, 1).contiguous()
                output = model(input_).permute(0, 3, 1, 2).contiguous()
                loss = nn.MSELoss()(output, truth)

                # Record and print loss
                val_loss_epoch += loss.item()
                if (i + 1) % args.display_interval == 0:
                    print('Epoch: [{}][{}]\tBatch: [{}][{}]\tLoss: {:.4f}\tTime: {:.4f}'.format(
                        epoch + 1, total_epochs, i + 1, len(val_loader), loss.item(), 
                        time.time() - val_batch_timer))
                    val_batch_timer = time.time()
            
        # Save val loss
        val_loss_epoch = val_loss_epoch / len(val_loader)
        print('Epoch: [{}][{}]\tLoss: {:.4f}\tTime: {:.4f}'.format(
            epoch + 1, total_epochs, val_loss_epoch, time.time() - val_epoch_timer))
        val_epoch_timer = time.time()
        val_loss.append(val_loss_epoch)
        np.savetxt(os.path.join(args.output_path, 'val_loss.txt'), val_loss)
        print('Val loss saved')

        # Plot loss
        visualizer.plot_loss(train_loss, val_loss, args.output_path, 'loss.jpg')
        print('Loss figure saved')

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
        save_checkpoint(checkpoint_path, current_iteration, train_loss, val_loss, model, optimizer)
        if args.early_stopping:
            early_stopping_flag = early_stopping(val_loss)
            if early_stopping_flag:
                print('Early stopped')
                break


@torch.no_grad()
def test(model: nn.Module, test_loader: DataLoader):
    metrics = {}
    metrics['MAE'] = 0
    metrics['RMSE'] = 0
    metrics['MBE'] = 0

    # Timer
    test_timer = time.time()
    test_batch_timer = time.time()

    # Test
    print('\n[Test]')
    bestparams_path = os.path.join(args.output_path, 'bestparams.pth')
    states = load_checkpoint(bestparams_path, args.device)
    model.load_state_dict(states['model'])
    model.eval()
    
    for i, (t, elev, ref) in enumerate(test_loader):
        # Forward propagation
        ref = ref.to(args.device)
        input_, truth = ref[:, 1:], ref[:, :1]
        input_ = input_.permute(0, 2, 3, 1).contiguous()
        output = model(input_).permute(0, 3, 1, 2).contiguous()
        masked_ref, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
            ref, args.azimuth_blockage_range, args.random_seed + i)
        output = masked_ref[:, :1] + output * (1 - mask[:, :1])

        # Print time
        if (i + 1) % args.display_interval == 0:
            print('Batch: [{}][{}]\tTime: {:.4f}'.format(
                i + 1, len(test_loader), time.time() - test_batch_timer))
            test_batch_timer = time.time()

        # Evaluation
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
    model.load_state_dict(states['model'])
    model.eval()
    for i, (t, elev, ref) in enumerate(case_loader):
        print('\nCase {} at {}'.format(i, t))
        metrics = {}

        # Forward propagation
        ref = ref.to(args.device)
        input_, truth = ref[:, 1:], ref[:, :1]
        input_ = input_.permute(0, 2, 3, 1).contiguous()
        output = model(input_).permute(0, 3, 1, 2).contiguous()
        masked_ref, mask, anchor, blockage_len = maskutils.gen_fixed_blockage_mask(
            ref, args.azimuthal_range[0], args.case_anchor[i], args.case_blockage_len[i])
        output = masked_ref[:, :1] + output * (1 - mask[:, :1])

        # Evaluation
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
