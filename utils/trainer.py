import os
import math
import numpy as np
import pandas as pd
import torch
from torch.optim import Adadelta

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

    def fit(self, generator, discriminator, train_loader, val_loader, test_loader):
        self.generator = generator
        self.discriminator = discriminator
        self.generator.to(self.args.device)
        self.discriminator.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        if self.args.pretrain:
            start_iterations = self.load_checkpoint()['iteration']
        else:
            start_iterations = 0
        self.total_epochs = int(math.ceil((self.args.max_iterations - start_iterations) / len(train_loader)))
        
        self.optimizer_g = Adadelta(self.generator.parameters())
        self.optimizer_d = Adadelta(self.discriminator.parameters())

        if self.args.train:
            self.train()
        if self.args.test:
            self.test()
    
    def train(self):
        # Pretrain: Load model and optimizer
        if self.args.pretrain:
            states = self.load_checkpoint()
            self.generator.load_state_dict(states['generator'])
            self.discriminator.load_state_dict(states['discriminator'])
            self.optimizer_g.load_state_dict(states['optimizer_g'])
            self.optimizer_d.load_state_dict(states['optimizer_d'])
            self.current_iterations = states['iteration']
            self.train_loss_g = states['train_loss_g']
            self.train_loss_d = states['train_loss_d']
            self.val_loss_g = states['val_loss_g']
            self.val_loss_d = states['val_loss_d']
        else:
            self.current_iterations = 0
            self.train_loss_g = []
            self.train_loss_d = []
            self.val_loss_g = []
            self.val_loss_d = []

        early_stopping = EarlyStopping(verbose=True, path='bestmodel.pt')

        for epoch in range(self.total_epochs):
            print('\n[Train]')
            print('Epoch: [{}][{}]'.format(epoch + 1, self.total_epochs))
            train_loss_g = []
            train_loss_d = []
            val_loss_g = []
            val_loss_d = []

            # Train
            self.generator.train()
            self.discriminator.train()

            for i, tensor in enumerate(self.train_loader):
                # load data
                tensor = tensor.to(self.args.device)
                tensor = scaler.minmax_norm(tensor, self.args.vmax, self.args.vmin)
                hole_area_fake = maskutils.gen_hole_area(
                    (self.args.ld_input_size, self.args.ld_input_size),
                    (tensor.shape[3], tensor.shape[2]))
                mask = maskutils.gen_input_mask(
                    shape=(tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]),
                    hole_size=(
                        (self.args.hole_min_w, self.args.hole_max_w),
                        (self.args.hole_min_h, self.args.hole_max_h)),
                    hole_area=hole_area_fake).to(self.args.device)
                tensor_masked = tensor - tensor * mask
                
                # discriminator fake forward
                input_ = torch.cat((tensor_masked, mask), dim=1)
                output = self.generator(input_)
                input_gd_fake = output.detach()
                input_ld_fake = maskutils.crop(input_gd_fake, hole_area_fake)
                output_fake = self.discriminator((input_ld_fake, input_gd_fake))
                
                # discriminator real forward
                hole_area_real = maskutils.gen_hole_area(
                    (self.args.ld_input_size, self.args.ld_input_size),
                    (tensor.shape[3], tensor.shape[2]))
                input_gd_real = tensor
                input_ld_real = maskutils.crop(input_gd_real, hole_area_real)
                output_real = self.discriminator((input_ld_real, input_gd_real))
                
                # discriminator loss
                loss_d = losses.cal_d_loss(output_fake, output_real)
                
                # discriminator backward
                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

                # generator forward
                loss_g_1 = losses.completion_network_loss(tensor, output, mask)
                input_gd_fake = output
                input_ld_fake = maskutils.crop(input_gd_fake, hole_area_fake)
                output_fake = self.discriminator((input_ld_fake, input_gd_fake))
                loss_g_2 = losses.cal_g_loss(output_fake)

                # generator loss
                loss_g = (loss_g_1 + self.args.alpha * loss_g_2) / 2
                
                # generator backward
                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()
   
                # back scaling
                tensor = scaler.reverse_minmax_norm(tensor, self.args.vmax, self.args.vmin)
                tensor_masked = scaler.reverse_minmax_norm(tensor_masked, self.args.vmax, self.args.vmin)
                output = scaler.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

                train_loss_g.append(loss_g.item())
                train_loss_d.append(loss_d.item())
                if (i + 1) % self.args.display_interval == 0:
                    print('Epoch: [{}][{}] Batch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                        epoch + 1, self.total_epochs, i + 1, len(self.train_loader), loss_g.item(), loss_d.item()
                    ))
            
            self.train_loss_g.append(np.mean(train_loss_g))
            self.train_loss_d.append(np.mean(train_loss_d))
            np.savetxt(os.path.join(self.args.output_path, 'train_loss_g.txt'), self.train_loss_g)
            np.savetxt(os.path.join(self.args.output_path, 'train_loss_d.txt'), self.train_loss_d)
            
            tensors = torch.cat([tensor, tensor_masked, output], dim=1)
            visualizer.plot_map(tensors, self.args.output_path, 'train')
            print('Epoch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                epoch + 1, self.total_epochs, self.train_loss_g[-1], self.train_loss_d[-1]))

            # Validate
            print('\n[Val]')
            self.generator.eval()
            self.discriminator.eval()
            
            with torch.no_grad():
                for i, tensor in enumerate(self.val_loader):
                    # load data
                    tensor = tensor.to(self.args.device)
                    tensor = scaler.minmax_norm(tensor, self.args.vmax, self.args.vmin)
                    hole_area_fake = maskutils.gen_hole_area(
                        (self.args.ld_input_size, self.args.ld_input_size),
                        (tensor.shape[3], tensor.shape[2]))
                    mask = maskutils.gen_input_mask(
                        shape=(tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]),
                        hole_size=(
                            (self.args.hole_min_w, self.args.hole_max_w),
                            (self.args.hole_min_h, self.args.hole_max_h)),
                        hole_area=hole_area_fake).to(self.args.device)
                    tensor_masked = tensor - tensor * mask

                    # discriminator fake forward
                    input_ = torch.cat((tensor_masked, mask), dim=1)
                    output = self.generator(input_)
                    input_gd_fake = output.detach()
                    input_ld_fake = maskutils.crop(input_gd_fake, hole_area_fake)
                    output_fake = self.discriminator((input_ld_fake, input_gd_fake))
                    
                    # discriminator real forward
                    hole_area_real = maskutils.gen_hole_area(
                        (self.args.ld_input_size, self.args.ld_input_size),
                        (tensor.shape[3], tensor.shape[2]))
                    input_gd_real = tensor
                    input_ld_real = maskutils.crop(input_gd_real, hole_area_real)
                    output_real = self.discriminator((input_ld_real, input_gd_real)) 
                    
                    # discriminator loss
                    loss_d = losses.cal_d_loss(output_fake, output_real)

                    # generator forward
                    loss_g_1 = losses.completion_network_loss(tensor, output, mask)
                    input_gd_fake = output
                    input_ld_fake = maskutils.crop(input_gd_fake, hole_area_fake)
                    output_fake = self.discriminator((input_ld_fake, (input_gd_fake)))
                    loss_g_2 = losses.cal_g_loss(output_fake)

                    # generator loss
                    loss_g = (loss_g_1 + self.args.alpha * loss_g_2) / 2

                    # back scaling
                    tensor = scaler.reverse_minmax_norm(tensor, self.args.vmax, self.args.vmin)
                    tensor_masked = scaler.reverse_minmax_norm(tensor_masked, self.args.vmax, self.args.vmin)
                    output = scaler.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

                    val_loss_g.append(loss_g.item())
                    val_loss_d.append(loss_d.item())
                    if (i + 1) % self.args.display_interval == 0:
                        print('Epoch: [{}][{}] Batch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                            epoch + 1, self.total_epochs, i + 1, len(self.val_loader), loss_g.item(), loss_d.item()
                        ))
                            
            self.val_loss_g.append(np.mean(val_loss_g))
            self.val_loss_d.append(np.mean(val_loss_d))            
            np.savetxt(os.path.join(self.args.output_path, 'val_loss_g.txt'), self.val_loss_g)
            np.savetxt(os.path.join(self.args.output_path, 'val_loss_d.txt'), self.val_loss_d)
            
            tensors = torch.cat([tensor, tensor_masked, output], dim=1)
            visualizer.plot_map(tensors, self.args.output_path, 'val')
            print('Epoch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                epoch + 1, self.total_epochs, self.val_loss_g[-1], self.val_loss_d[-1]))

            visualizer.plot_loss(self.train_loss_g, self.val_loss_g, self.args.output_path, 'loss_g.png')
            visualizer.plot_loss(self.train_loss_d, self.val_loss_d, self.args.output_path, 'loss_d.png')

            # Save checkpoint
            self.save_checkpoint()

            if self.args.early_stopping:
                early_stopping(self.val_loss_g[-1], self)
            
            if early_stopping.early_stop:
                break
            
            self.current_iterations += 1
            if self.current_iterations == self.args.max_iterations:
                print('Max interations %d reached.' % self.args.max_iterations)
                break

    def test(self):
        metrics = {}
        metrics['MAE'] = []
        metrics['RMSE'] = []
        metrics['COSSIM'] = []
        metrics['SSIM'] = []
        metrics['PSNR'] = []
        
        print('\n[Test]')
        self.generator.load_state_dict(self.load_checkpoint('bestmodel.pt')['generator'])
        self.generator.eval()
        
        with torch.no_grad():
            for i, tensor in enumerate(self.test_loader):
                tensor = tensor.to(self.args.device)
                tensor = scaler.minmax_norm(tensor, self.args.vmax, self.args.vmin)
                hole_area_fake = maskutils.gen_hole_area(
                    (self.args.ld_input_size, self.args.ld_input_size),
                    (tensor.shape[3], tensor.shape[2]))
                mask = maskutils.gen_input_mask(
                    shape=(tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]),
                    hole_size=(
                        (self.args.hole_min_w, self.args.hole_max_w),
                        (self.args.hole_min_h, self.args.hole_max_h)),
                    hole_area=hole_area_fake).to(self.args.device)
                tensor_masked = tensor - tensor * mask

                # discriminator fake forward
                input_ = torch.cat((tensor_masked, mask), dim=1)
                output = self.generator(input_)

                # back scaling
                tensor = scaler.reverse_minmax_norm(tensor, self.args.vmax, self.args.vmin)
                tensor_masked = scaler.reverse_minmax_norm(tensor_masked, self.args.vmax, self.args.vmin)
                output = scaler.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

                if (i + 1) % self.args.display_interval == 0:
                    print('Batch: [{}][{}]'.format(i + 1, len(self.test_loader)))

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

        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'test_metrics.csv'), float_format='%.8f', index=False)
        
        tensors = torch.cat([tensor, tensor_masked, output], dim=1)
        visualizer.plot_map(tensors, self.args.output_path, 'test')
        
        print('Test done.')

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
        
        with torch.no_grad():
            for tensor in sample_loader:
                tensor = tensor.to(self.args.device)
                tensor = scaler.minmax_norm(tensor, self.args.vmax, self.args.vmin)
                hole_area_fake = maskutils.gen_hole_area(
                    (self.args.ld_input_size, self.args.ld_input_size),
                    (tensor.shape[3], tensor.shape[2]))
                mask = maskutils.gen_input_mask(
                    shape=(tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]),
                    hole_size=(
                        (self.args.hole_min_w, self.args.hole_max_w),
                        (self.args.hole_min_h, self.args.hole_max_h)),
                    hole_area=hole_area_fake).to(self.args.device)
                tensor_masked = tensor - tensor * mask

                # discriminator fake forward
                input_ = torch.cat((tensor_masked, mask), dim=1)
                output = generator(input_)

                # back scaling
                tensor = scaler.reverse_minmax_norm(tensor, self.args.vmax, self.args.vmin)
                tensor_masked = scaler.reverse_minmax_norm(tensor_masked, self.args.vmax, self.args.vmin)
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

        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'predict_metrics.csv'), float_format='%.8f', index=False)
        
        tensors = torch.cat([tensor, tensor_masked, output], dim=1)
        visualizer.plot_map(tensors, self.args.output_path, 'predict')
        
        print('Predict done.')

    def save_checkpoint(self, filename='checkpoint.pt'):
        states = {
            'iteration': self.current_iterations,
            'train_loss_g': self.train_loss_g,
            'train_loss_d': self.train_loss_d,
            'val_loss_g': self.val_loss_g,
            'val_loss_d': self.val_loss_d,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict()
        }
        torch.save(states, os.path.join(self.args.output_path, filename))

    def load_checkpoint(self, filename='checkpoint.pt'):
        states = torch.load(os.path.join(self.args.output_path, filename), map_location=self.args.device)
        return states
