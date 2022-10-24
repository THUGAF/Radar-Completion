import os
import math
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.scaler as scaler
import utils.maskutils as maskutils
import model.losses as losses


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'       
        """
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, trainer):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, trainer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, trainer)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, trainer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        trainer.save_checkpoint(self.path)
        self.val_loss_min = val_loss


class Trainer:
    def __init__(self, args):
        self.args = args

    def fit(self, model, train_loader, val_loader, test_loader):
        self.model = model.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        if self.args.pretrain:
            start_iterations = self.load_checkpoint()['iteration']
        else:
            start_iterations = 0
        self.total_epochs = int(math.ceil((self.args.max_iterations - start_iterations) / len(train_loader)))

        self.optimizer = Adam(self.model.parameters())

        if self.args.train:
            self.train()
        if self.args.test:
            self.test()
    
    def train(self):
        # Pretrain: Load model and optimizer
        if self.args.pretrain:
            states = self.load_checkpoint()
            self.model.load_state_dict(states['model'])
            self.optimizer.load_state_dict(states['optimizer_g'])
            self.current_iterations = states['iteration']
            self.train_loss = states['train_loss']
            self.val_loss = states['val_loss']
        else:
            self.current_iterations = 0
            self.train_loss = []
            self.val_loss = []
        early_stopping = EarlyStopping(verbose=True, path='bestmodel.pt')

        # Train
        for epoch in range(self.total_epochs):
            print('\n[Train]')
            print('Epoch: [{}][{}]'.format(epoch + 1, self.total_epochs))
            train_loss = []
            val_loss = []

            # Train
            self.model.train()

            for i, (elevs, tensor) in enumerate(self.train_loader):
                # load data
                tensor = tensor.to(self.args.device)
                tensor = scaler.minmax_norm(tensor, self.args.vmax, self.args.vmin)
                masked_tensor, mask, anchor, blockage_len = maskutils.gen_blockage_mask(tensor, 
                    self.args.azimuth_blockage_range, self.args.random_seed + i)
                
                # forward
                input_ = torch.cat([masked_tensor, mask], dim=1)
                output = self.model(input_)
                output = masked_tensor[:, :1] + output * (1 - mask[:, :1])
                loss = losses.biased_mae_loss(output, tensor[:, :1], self.args.vmax, self.args.vmin)
                
                # discriminator backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
   
                # back scaling
                tensor = scaler.reverse_minmax_norm(tensor, self.args.vmax, self.args.vmin)
                masked_tensor = scaler.reverse_minmax_norm(masked_tensor, self.args.vmax, self.args.vmin)
                output = scaler.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

                train_loss.append(loss.item())
                if (i + 1) % self.args.display_interval == 0:
                    print('Epoch: [{}][{}] Batch: [{}][{}] Loss: {:.6f}'.format(
                        epoch + 1, self.total_epochs, i + 1, len(self.train_loader), loss.item()))
            
            # Save loss
            self.train_loss.append(np.mean(train_loss))
            np.savetxt(os.path.join(self.args.output_path, 'train_loss.txt'), self.train_loss)
            print('Epoch: [{}][{}] Loss: {:.6f}'.format(epoch + 1, self.total_epochs, self.train_loss[-1]))
            
            # Plot tensors
            tensors = torch.cat([tensor, masked_tensor[:, :1], output], dim=1)
            visualizer.plot_tensors(tensors, self.args.azimuth_range[0], self.args.radial_range[0], 
                                    anchor, blockage_len, self.args.output_path, 'train')
            print('Tensors plotted.')

            # Validate
            print('\n[Val]')
            self.model.eval()

            with torch.no_grad():
                for i, (elevs, tensor) in enumerate(self.val_loader):
                    # load data
                    tensor = tensor.to(self.args.device)
                    tensor = scaler.minmax_norm(tensor, self.args.vmax, self.args.vmin)
                    masked_tensor, mask = maskutils.gen_blockage_mask(tensor, 
                        self.args.azimuth_blockage_range, self.args.random_seed + i)

                    # forward
                    input_ = torch.cat([masked_tensor, mask], dim=1)
                    output = self.model(input_)
                    output = masked_tensor[:, :1] + output * (1 - mask[:, :1])
                    loss = losses.biased_mae_loss(output, tensor[:, :1], self.args.vmax, self.args.vmin)

                    # back scaling
                    tensor = scaler.reverse_minmax_norm(tensor, self.args.vmax, self.args.vmin)
                    masked_tensor = scaler.reverse_minmax_norm(masked_tensor, self.args.vmax, self.args.vmin)
                    output = scaler.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

                    val_loss.append(loss.item())
                    if (i + 1) % self.args.display_interval == 0:
                        print('Epoch: [{}][{}] Batch: [{}][{}] Loss: {:.6f}'.format(
                            epoch + 1, self.total_epochs, i + 1, len(self.val_loader), loss.item()))
            
            # Save loss                
            self.val_loss.append(np.mean(val_loss))         
            np.savetxt(os.path.join(self.args.output_path, 'val_loss.txt'), self.val_loss)
            print('Epoch: [{}][{}] Loss: {:.6f}'.format(epoch + 1, self.total_epochs, self.val_loss[-1]))
            
            # Plot tensors
            tensors = torch.cat([tensor, masked_tensor[:, :1], output], dim=1)
            visualizer.plot_tensors(tensors, self.args.azimuth_range[0], self.args.radial_range[0],
                                    anchor, blockage_len, self.args.output_path, 'val')
            print('Tensors plotted.')
            
            # Plot loss
            visualizer.plot_loss(self.train_loss, self.val_loss, self.args.output_path, 'loss.png')

            # Save checkpoint
            self.save_checkpoint()

            # Check early stopping
            if self.args.early_stopping:
                early_stopping(self.val_loss[-1], self)
            if early_stopping.early_stop:
                break
            
            # Check max iterations
            self.current_iterations += 1
            if self.current_iterations == self.args.max_iterations:
                print('Max interations %d reached.' % self.args.max_iterations)
                break
    
    @torch.no_grad()
    def test(self):
        metrics = {}
        metrics['MAE'] = []
        metrics['RMSE'] = []
        metrics['COSSIM'] = []
        metrics['SSIM'] = []
        metrics['PSNR'] = []
        
        print('\n[Test]')
        self.model.load_state_dict(self.load_checkpoint('bestmodel.pt')['generator'])
        self.model.eval()
        
        for i, (elevs, tensor) in enumerate(self.test_loader):
            tensor = tensor.to(self.args.device)
            tensor = scaler.minmax_norm(tensor, self.args.vmax, self.args.vmin)
            masked_tensor, mask, anchor, blockage_len = maskutils.gen_blockage_mask(tensor, 
                self.args.azimuth_blockage_range, self.args.random_seed + i)
            
            # forward
            input_ = torch.cat([masked_tensor, mask], dim=1)
            output = self.model(input_)
            output = masked_tensor[:, :1] + output * (1 - mask[:, :1])

            # back scaling
            tensor = scaler.reverse_minmax_norm(tensor, self.args.vmax, self.args.vmin)
            masked_tensor = scaler.reverse_minmax_norm(masked_tensor, self.args.vmax, self.args.vmin)
            output = scaler.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

            if (i + 1) % self.args.display_interval == 0:
                print('Batch: [{}][{}]'.format(i + 1, len(self.test_loader)))

            # evaluation
            metrics['MAE'].append(evaluation.evaluate_mae(tensor, output))
            metrics['RMSE'].append(evaluation.evaluate_rmse(tensor, output))
            metrics['COSSIM'].append(evaluation.evaluate_cossim(tensor, output))
            metrics['SSIM'].append(evaluation.evaluate_ssim(tensor, output))
            metrics['PSNR'].append(evaluation.evaluate_psnr(tensor, output))

        metrics['MAE'].append(np.mean(metrics['MAE'], axis=0))
        metrics['RMSE'].append(np.mean(metrics['RMSE'], axis=0))
        metrics['COSSIM'].append(np.mean(metrics['COSSIM'], axis=0))
        metrics['SSIM'].append(np.mean(metrics['SSIM'], axis=0))
        metrics['PSNR'].append(np.mean(metrics['PSNR'], axis=0))

        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'test_metrics.csv'), float_format='%.8f', index=False)
        tensors = torch.cat([tensor, masked_tensor[:, :1], output], dim=1)
        visualizer.plot_tensors(tensors, self.args.azimuth_range[0], self.args.radial_range[0],
                                anchor, blockage_len, self.args.output_path, 'test')
        print('Test done.')

    @torch.no_grad()
    def predict(self, generator, sample_loader):
        metrics = {}
        metrics['MAE'] = []
        metrics['RMSE'] = []
        metrics['COSSIM'] = []
        metrics['SSIM'] = []
        metrics['PSNR'] = []
        
        print('\n[Predict]')
        generator.load_state_dict(self.load_checkpoint('bestmodel.pt')['generator'])
        generator.eval()
        
        for elevs, tensor in sample_loader:
            tensor = tensor.to(self.args.device)
            tensor = scaler.minmax_norm(tensor, self.args.vmax, self.args.vmin)
            masked_tensor, mask, anchor, blockage_len = maskutils.gen_blockage_mask(tensor, 
                self.args.azimuth_blockage_range, self.args.random_seed)
            
            # forward
            input_ = torch.cat([masked_tensor, mask], dim=1)
            output = self.model(input_)
            output = masked_tensor[:, :1] + output * (1 - mask[:, :1])

            # back scaling
            tensor = scaler.reverse_minmax_norm(tensor, self.args.vmax, self.args.vmin)
            masked_tensor = scaler.reverse_minmax_norm(masked_tensor, self.args.vmax, self.args.vmin)
            output = scaler.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

            # evaluation
            metrics['MAE'].append(evaluation.evaluate_mae(tensor, output))
            metrics['RMSE'].append(evaluation.evaluate_rmse(tensor, output))
            metrics['COSSIM'].append(evaluation.evaluate_cossim(tensor, output))
            metrics['SSIM'].append(evaluation.evaluate_ssim(tensor, output))
            metrics['PSNR'].append(evaluation.evaluate_psnr(tensor, output))

        metrics['MAE'] = np.mean(metrics['MAE'], axis=0)
        metrics['RMSE'] = np.mean(metrics['RMSE'], axis=0)
        metrics['COSSIM'] = np.mean(metrics['COSSIM'], axis=0)
        metrics['SSIM'] = np.mean(metrics['SSIM'], axis=0)
        metrics['PSNR'] = np.mean(metrics['PSNR'], axis=0)

        df = pd.DataFrame(data=metrics, index=['MAE'])
        df.to_csv(os.path.join(self.args.output_path, 'predict_metrics.csv'), float_format='%.8f', index=False)
        tensors = torch.cat([tensor, masked_tensor[:, :1], output], dim=1)
        visualizer.plot_tensors(tensors, self.args.azimuth_range[0], self.args.radial_range[0],
                                anchor, blockage_len, self.args.output_path, 'predict')
        print('Predict done.')

    def save_checkpoint(self, filename='checkpoint.pt'):
        states = {
            'iteration': self.current_iterations,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(states, os.path.join(self.args.output_path, filename))

    def load_checkpoint(self, filename='checkpoint.pt'):
        states = torch.load(os.path.join(self.args.output_path, filename), map_location=self.args.device)
        return states
