import os
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.scaler as scaler
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
        self.model = model
        self.model.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if self.args.pretrain:
            start_iterations = self.load_checkpoint()['iteration']
        else:
            start_iterations = 0
        self.total_epochs = int(math.ceil((self.args.max_iterations - start_iterations) / len(train_loader)))
        
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr,
                              betas=(self.args.beta1, self.args.beta2),
                              weight_decay=self.args.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=2000, gamma=0.5)

        if self.args.train:
            self.train()
        if self.args.test:
            self.test()

    def train(self):
        # Pretrain: Load model and optimizer
        if self.args.pretrain:
            states = self.load_checkpoint()
            self.model.load_state_dict(states['model'])
            self.optimizer.load_state_dict(states['optimizer'])
            self.current_iterations = states['iteration']
            self.train_loss = states['train_loss']
            self.val_loss = states['val_loss']
        else:
            self.current_iterations = 0
            self.train_loss = []
            self.val_loss = []

        early_stopping = EarlyStopping(verbose=True, path='bestmodel.pt')

        for epoch in range(self.total_epochs):
            print('\n[Train]')
            print('Epoch: [{}][{}]'.format(epoch + 1, self.total_epochs))
            train_loss = []
            val_loss = []

            # Train
            self.model.train()
            for i, (tensor, timestamp) in enumerate(self.train_loader):
                tensor = tensor.transpose(1, 0).to(self.args.device)
                timestamp = timestamp.transpose(1, 0).to(self.args.device)
                input_ = tensor[:self.args.input_steps]
                truth = tensor[self.args.input_steps:]
                input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)

                pred = self.model(input_)

                # Backward propagation
                loss = losses.biased_mae_loss(pred, truth, self.args.vmax) + self.args.var_reg * losses.cv_loss(pred, truth)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                input_ = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
                pred = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
                truth = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)

                train_loss.append(loss.item())
                if (i + 1) % self.args.display_interval == 0:
                    print('Epoch: [{}][{}] Batch: [{}][{}] Loss: {:.4f}'.format(
                        epoch + 1, self.total_epochs, i + 1, len(self.train_loader), loss.item()
                    ))
            
            self.train_loss.append(np.mean(train_loss))
            np.savetxt(os.path.join(self.args.output_path, 'train_loss.txt'), self.train_loss)
            visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'train')
            print('Epoch: [{}][{}] Loss: {:.4f}'.format(epoch + 1, self.total_epochs, self.train_loss[-1]))

            # Validate
            print('\n[Val]')
            self.model.eval()
            with torch.no_grad():
                for i, (tensor, timestamp) in enumerate(self.val_loader):
                    tensor = tensor.transpose(1, 0).to(self.args.device)
                    timestamp = timestamp.transpose(1, 0).to(self.args.device)
                    input_ = tensor[:self.args.input_steps]
                    truth = tensor[self.args.input_steps:]
                    input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                    truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)
                    
                    pred = self.model(input_)
                    loss = losses.biased_mae_loss(pred, truth, self.args.vmax) + self.args.var_reg * losses.cv_loss(pred, truth)

                    input_ = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
                    pred = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
                    truth = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)

                    val_loss.append(loss.item())
                    if (i + 1) % self.args.display_interval == 0:
                        print('Epoch: [{}][{}] Batch: [{}][{}] Loss: {:.4f}'.format(
                            epoch + 1, self.total_epochs, i + 1, len(self.val_loader), loss.item()
                        ))
                            
            self.val_loss.append(np.mean(val_loss))            
            np.savetxt(os.path.join(self.args.output_path, 'val_loss.txt'), self.val_loss)
            visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'val')
            print('Epoch: [{}][{}] Loss: {:.4f}'.format(epoch + 1, self.total_epochs, self.val_loss[-1]))

            visualizer.plot_loss(self.train_loss, self.val_loss, self.args.output_path)

            # Save checkpoint
            self.save_checkpoint()

            if self.args.early_stopping:
                early_stopping(self.val_loss[-1], self)
            
            if early_stopping.early_stop:
                break
            
            self.current_iterations += 1
            if self.current_iterations == self.args.max_iterations:
                print('Max interations %d reached.' % self.args.max_iterations)
                break

    def test(self):
        metrics = {}
        metrics['Time'] = np.linspace(6, 60, 10)
        for threshold in self.args.thresholds:
            metrics['POD-%ddBZ' % threshold] = []
            metrics['FAR-%ddBZ' % threshold] = []
            metrics['CSI-%ddBZ' % threshold] = []
            metrics['HSS-%ddBZ' % threshold] = []
        metrics['CC'] = []
        metrics['ME'] = []
        metrics['MAE'] = []
        metrics['RMSE'] = []
        metrics['SSIM'] = []
        metrics['PSNR'] = []
        metrics['CVR'] = []
        metrics['SSDR'] = []
        test_loss = []
        
        self.model.load_state_dict(self.load_checkpoint('bestmodel.pt')['model'])
        self.model.eval()
        print('\n[Test]')
        with torch.no_grad():
            for i, (tensor, timestamp) in enumerate(self.test_loader):
                tensor = tensor.transpose(1, 0).to(self.args.device)
                timestamp = timestamp.transpose(1, 0).to(self.args.device)
                input_ = tensor[:self.args.input_steps]
                truth = tensor[self.args.input_steps:]
                input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)

                pred = self.model(input_)
                loss = losses.biased_mae_loss(pred, truth, self.args.vmax) + self.args.var_reg * losses.cv_loss(pred, truth)

                input_rev = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
                pred_rev = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
                truth_rev = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)

                test_loss.append(loss.item())
                if (i + 1) % self.args.display_interval == 0:
                    print('Batch: [{}][{}] Loss: {:.4f}'.format(i + 1, len(self.test_loader), loss.item()))

                for threshold in self.args.thresholds:
                    pod, far, csi, hss = evaluation.evaluate_forecast(pred_rev, truth_rev, threshold)
                    metrics['POD-%ddBZ' % threshold].append(pod)
                    metrics['FAR-%ddBZ' % threshold].append(far)
                    metrics['CSI-%ddBZ' % threshold].append(csi)
                    metrics['HSS-%ddBZ' % threshold].append(hss)
                metrics['CC'].append(evaluation.evaluate_cc(pred_rev, truth_rev))
                metrics['ME'].append(evaluation.evaluate_me(pred_rev, truth_rev))
                metrics['MAE'].append(evaluation.evaluate_mae(pred_rev, truth_rev))
                metrics['RMSE'].append(evaluation.evaluate_rmse(pred_rev, truth_rev))
                metrics['SSIM'].append(evaluation.evaluate_ssim(pred, truth))
                metrics['PSNR'].append(evaluation.evaluate_psnr(pred, truth))
                metrics['CVR'].append(evaluation.evaluate_cvr(pred, truth))
                metrics['SSDR'].append(evaluation.evaluate_ssdr(pred, truth))
                    
        print('Loss: {:.4f}'.format(np.mean(test_loss)))
        for threshold in self.args.thresholds:
            metrics['POD-%ddBZ' % threshold] = np.mean(metrics['POD-%ddBZ' % threshold], axis=0)
            metrics['FAR-%ddBZ' % threshold] = np.mean(metrics['FAR-%ddBZ' % threshold], axis=0)
            metrics['CSI-%ddBZ' % threshold] = np.mean(metrics['CSI-%ddBZ' % threshold], axis=0)
            metrics['HSS-%ddBZ' % threshold] = np.mean(metrics['HSS-%ddBZ' % threshold], axis=0)
        metrics['CC'] = np.mean(metrics['CC'], axis=0)
        metrics['ME'] = np.mean(metrics['ME'], axis=0)
        metrics['MAE'] = np.mean(metrics['MAE'], axis=0)
        metrics['RMSE'] = np.mean(metrics['RMSE'], axis=0)
        metrics['SSIM'] = np.mean(metrics['SSIM'], axis=0)
        metrics['PSNR'] = np.mean(metrics['PSNR'], axis=0)
        metrics['CVR'] = np.mean(metrics['CVR'], axis=0)
        metrics['SSDR'] = np.mean(metrics['SSDR'], axis=0)

        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'test_metrics.csv'), float_format='%.8f', index=False)
        visualizer.plot_map(input_rev, pred_rev, truth_rev, timestamp, self.args.output_path, 'test')
        print('Test done.')

    def predict(self, model, sample_loader):
        metrics = {}
        metrics['Time'] = np.linspace(6, 60, 10)
        self.model = model
        self.model.to(self.args.device)
        self.model.load_state_dict(self.load_checkpoint('bestmodel.pt')['model'])
        self.model.eval()
        print('\n[Predict]')
        with torch.no_grad():
            for i, (tensor, timestamp) in enumerate(sample_loader):
                tensor = tensor.transpose(1, 0).to(self.args.device)
                timestamp = timestamp.transpose(1, 0).to(self.args.device)
                input_ = tensor[:self.args.input_steps]
                truth = tensor[self.args.input_steps:]
                input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)
                
                pred = self.model(input_)
                input_rev = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
                pred_rev = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
                truth_rev = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)

                for threshold in self.args.thresholds:
                    pod, far, csi, hss = evaluation.evaluate_forecast(pred_rev, truth_rev, threshold)
                    metrics['POD-%ddBZ' % threshold] = pod
                    metrics['FAR-%ddBZ' % threshold] = far
                    metrics['CSI-%ddBZ' % threshold] = csi
                    metrics['HSS-%ddBZ' % threshold] = hss

                metrics['CC'] = evaluation.evaluate_cc(pred_rev, truth_rev)
                metrics['ME'] = evaluation.evaluate_me(pred_rev, truth_rev)
                metrics['MAE'] = evaluation.evaluate_mae(pred_rev, truth_rev)
                metrics['RMSE'] = evaluation.evaluate_rmse(pred_rev, truth_rev)
                metrics['SSIM'] = evaluation.evaluate_ssim(pred, truth)
                metrics['PSNR'] = evaluation.evaluate_psnr(pred, truth)
                metrics['CVR'] = evaluation.evaluate_cvr(pred_rev, truth_rev)
                metrics['SSDR'] = evaluation.evaluate_ssdr(pred_rev, truth_rev)
                    
        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'sample_metrics.csv'), float_format='%.8f', index=False)
        visualizer.plot_map(input_rev, pred_rev, truth_rev, timestamp, self.args.output_path, 'sample')
        print('Predict done.')

        return pred_rev
    
    def nowcast(self, model, input_loader):
        self.model = model
        self.model.to(self.args.device)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)['model'])
        self.model.eval()
        with torch.no_grad():
            for tensor, _ in input_loader:
                input_ = tensor.transpose(1, 0).to(self.args.device)
                input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                pred = self.model(input_)
                pred = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
        
        return pred

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


class GANTrainer:
    def __init__(self, args):
        self.args = args

    def fit(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.model.generator.to(self.args.device)
        self.model.discriminator.to(self.args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if self.args.pretrain:
            start_iterations = self.load_checkpoint()['iteration']
        else:
            start_iterations = 0
        self.total_epochs = int(math.ceil((self.args.max_iterations - start_iterations) / len(train_loader)))
        
        self.optimizer_g = Adam(self.model.generator.parameters(), lr=self.args.lr,
                                betas=(self.args.beta1, self.args.beta2),
                                weight_decay=self.args.weight_decay)
        self.optimizer_d = Adam(self.model.discriminator.parameters(), lr=self.args.lr * 2,
                                betas=(self.args.beta1, self.args.beta2),
                                weight_decay=self.args.weight_decay)
        self.scheduler_g = StepLR(self.optimizer_g, step_size=2000, gamma=0.5)
        self.scheduler_d = StepLR(self.optimizer_d, step_size=2000, gamma=0.5)

        if self.args.train:
            self.train()
        if self.args.test:
            self.test()
    
    def train(self):
        # Pretrain: Load model and optimizer
        if self.args.pretrain:
            states = self.load_checkpoint()
            self.model.generator.load_state_dict(states['model'])
            self.model.discriminator.load_state_dict(states['discriminator'])
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
            self.model.train()
            for i, (tensor, timestamp) in enumerate(self.train_loader):
                tensor = tensor.transpose(1, 0).to(self.args.device)
                timestamp = timestamp.transpose(1, 0).to(self.args.device)
                input_ = tensor[:self.args.input_steps]
                truth = tensor[self.args.input_steps:]
                input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)

                preds = [self.model(input_) for _ in range(self.args.ensemble_members)]
                real_score = self.model.discriminator(torch.cat([input_, truth]))
                fake_scores = [self.model.discriminator(torch.cat([input_, pred.detach()])) for pred in preds]
                
                # Backward propagation
                fake_score = torch.mean(torch.stack(fake_scores), dim=0)
                loss_d = losses.cal_d_loss(fake_score, real_score) * self.args.gan_reg
                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()
                self.scheduler_d.step()

                fake_scores = [self.model.discriminator(torch.cat([input_, pred])) for pred in preds]
                fake_score = torch.mean(torch.stack(fake_scores), dim=0)
                pred = torch.mean(torch.stack(preds), dim=0)
                loss_g = losses.biased_mae_loss(pred, truth, self.args.vmax) + \
                    losses.cv_loss(pred, truth) * self.args.var_reg + \
                    losses.cal_g_loss(fake_score) * self.args.gan_reg
                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()
                self.scheduler_g.step()
   
                input_ = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
                pred = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
                truth = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)

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
            visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'train')
            print('Epoch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                epoch + 1, self.total_epochs, self.train_loss_g[-1], self.train_loss_d[-1]))

            # Validate
            print('\n[Val]')
            self.model.eval()
            with torch.no_grad():
                for i, (tensor, timestamp) in enumerate(self.val_loader):
                    tensor = tensor.transpose(1, 0).to(self.args.device)
                    timestamp = timestamp.transpose(1, 0).to(self.args.device)
                    input_ = tensor[:self.args.input_steps]
                    truth = tensor[self.args.input_steps:]
                    input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                    truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)
                    
                    preds = [self.model(input_) for _ in range(self.args.ensemble_members)]
                    real_score = self.model.discriminator(torch.cat([input_, truth]))
                    fake_scores = [self.model.discriminator(torch.cat([input_, pred])) for pred in preds]
                    fake_score = torch.mean(torch.stack(fake_scores), dim=0)
                    loss_d = losses.cal_d_loss(fake_score, real_score) * self.args.gan_reg

                    pred = torch.mean(torch.stack(preds), dim=0)
                    loss_g = losses.biased_mae_loss(pred, truth, self.args.vmax) + \
                        losses.cv_loss(pred, truth) * self.args.var_reg + \
                        losses.cal_g_loss(fake_score) * self.args.gan_reg

                    input_ = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
                    pred = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
                    truth = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)

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
            visualizer.plot_map(input_, pred, truth, timestamp, self.args.output_path, 'val')
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
        metrics['Time'] = np.linspace(6, 60, 10)
        for threshold in self.args.thresholds:
            metrics['POD-%ddBZ' % threshold] = []
            metrics['FAR-%ddBZ' % threshold] = []
            metrics['CSI-%ddBZ' % threshold] = []
            metrics['HSS-%ddBZ' % threshold] = []
        metrics['CC'] = []
        metrics['ME'] = []
        metrics['MAE'] = []
        metrics['RMSE'] = []
        metrics['SSIM'] = []
        metrics['PSNR'] = []
        metrics['CVR'] = []
        metrics['SSDR'] = []
        test_loss_g = []
        test_loss_d = []
        
        self.model.generator.load_state_dict(self.load_checkpoint('bestmodel.pt')['model'])
        self.model.eval()
        print('\n[Test]')
        with torch.no_grad():
            for i, (tensor, timestamp) in enumerate(self.test_loader):
                tensor = tensor.transpose(1, 0).to(self.args.device)
                timestamp = timestamp.transpose(1, 0).to(self.args.device)
                input_ = tensor[:self.args.input_steps]
                truth = tensor[self.args.input_steps:]
                input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)

                preds = [self.model(input_) for _ in range(self.args.ensemble_members)]
                real_score = self.model.discriminator(torch.cat([input_, truth]))
                fake_scores = [self.model.discriminator(torch.cat([input_, pred])) for pred in preds]
                fake_score = torch.mean(torch.stack(fake_scores), dim=0)
                loss_d = losses.cal_d_loss(fake_score, real_score) * self.args.gan_reg
                
                pred = torch.mean(torch.stack(preds), dim=0)
                loss_g = losses.biased_mae_loss(pred, truth, self.args.vmax) + \
                    losses.cv_loss(pred, truth) * self.args.var_reg + \
                    losses.cal_g_loss(fake_score) * self.args.gan_reg

                input_rev = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
                pred_rev = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
                truth_rev = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)

                test_loss_g.append(loss_g.item())
                test_loss_d.append(loss_d.item())
                if (i + 1) % self.args.display_interval == 0:
                    print('Batch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                        i + 1, len(self.test_loader), loss_g.item(), loss_d.item()))

                for threshold in self.args.thresholds:
                    pod, far, csi, hss = evaluation.evaluate_forecast(pred_rev, truth_rev, threshold)
                    metrics['POD-%ddBZ' % threshold].append(pod)
                    metrics['FAR-%ddBZ' % threshold].append(far)
                    metrics['CSI-%ddBZ' % threshold].append(csi)
                    metrics['HSS-%ddBZ' % threshold].append(hss)
                metrics['CC'].append(evaluation.evaluate_cc(pred_rev, truth_rev))
                metrics['ME'].append(evaluation.evaluate_me(pred_rev, truth_rev))
                metrics['MAE'].append(evaluation.evaluate_mae(pred_rev, truth_rev))
                metrics['RMSE'].append(evaluation.evaluate_rmse(pred_rev, truth_rev))
                metrics['SSIM'].append(evaluation.evaluate_ssim(pred, truth))
                metrics['PSNR'].append(evaluation.evaluate_psnr(pred, truth))
                metrics['CVR'].append(evaluation.evaluate_cvr(pred, truth))
                metrics['SSDR'].append(evaluation.evaluate_ssdr(pred, truth))
                    
        print('Loss G: {:.4f} Loss D: {:.4f}'.format(np.mean(test_loss_g), np.mean(test_loss_d)))
        for threshold in self.args.thresholds:
            metrics['POD-%ddBZ' % threshold] = np.mean(metrics['POD-%ddBZ' % threshold], axis=0)
            metrics['FAR-%ddBZ' % threshold] = np.mean(metrics['FAR-%ddBZ' % threshold], axis=0)
            metrics['CSI-%ddBZ' % threshold] = np.mean(metrics['CSI-%ddBZ' % threshold], axis=0)
            metrics['HSS-%ddBZ' % threshold] = np.mean(metrics['HSS-%ddBZ' % threshold], axis=0)
        metrics['CC'] = np.mean(metrics['CC'], axis=0)
        metrics['ME'] = np.mean(metrics['ME'], axis=0)
        metrics['MAE'] = np.mean(metrics['MAE'], axis=0)
        metrics['RMSE'] = np.mean(metrics['RMSE'], axis=0)
        metrics['SSIM'] = np.mean(metrics['SSIM'], axis=0)
        metrics['PSNR'] = np.mean(metrics['PSNR'], axis=0)
        metrics['CVR'] = np.mean(metrics['CVR'], axis=0)
        metrics['SSDR'] = np.mean(metrics['SSDR'], axis=0)

        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'test_metrics.csv'), float_format='%.8f', index=False)
        visualizer.plot_map(input_rev, pred_rev, truth_rev, timestamp, self.args.output_path, 'test')
        print('Test done.')

    def predict(self, model, sample_loader):
        metrics = {}
        metrics['Time'] = np.linspace(6, 60, 10)
        self.model = model
        self.model.generator.to(self.args.device)
        self.model.generator.load_state_dict(self.load_checkpoint('bestmodel.pt')['model'])
        self.model.eval()
        print('\n[Predict]')
        with torch.no_grad():
            for i, (tensor, timestamp) in enumerate(sample_loader):
                tensor = tensor.transpose(1, 0).to(self.args.device)
                timestamp = timestamp.transpose(1, 0).to(self.args.device)
                input_ = tensor[:self.args.input_steps]
                truth = tensor[self.args.input_steps:]
                input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                truth = scaler.minmax_norm(truth, self.args.vmax, self.args.vmin)
                
                preds = [self.model(input_) for _ in range(self.args.ensemble_members)]
                pred = torch.mean(torch.stack(preds), dim=0)
                input_rev = scaler.reverse_minmax_norm(input_, self.args.vmax, self.args.vmin)
                pred_rev = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
                truth_rev = scaler.reverse_minmax_norm(truth, self.args.vmax, self.args.vmin)

                for threshold in self.args.thresholds:
                    pod, far, csi, hss = evaluation.evaluate_forecast(pred_rev, truth_rev, threshold)
                    metrics['POD-%ddBZ' % threshold] = pod
                    metrics['FAR-%ddBZ' % threshold] = far
                    metrics['CSI-%ddBZ' % threshold] = csi
                    metrics['HSS-%ddBZ' % threshold] = hss

                metrics['CC'] = evaluation.evaluate_cc(pred_rev, truth_rev)
                metrics['ME'] = evaluation.evaluate_me(pred_rev, truth_rev)
                metrics['MAE'] = evaluation.evaluate_mae(pred_rev, truth_rev)
                metrics['RMSE'] = evaluation.evaluate_rmse(pred_rev, truth_rev)
                metrics['SSIM'] = evaluation.evaluate_ssim(pred, truth)
                metrics['PSNR'] = evaluation.evaluate_psnr(pred, truth)
                metrics['CVR'] = evaluation.evaluate_cvr(pred_rev, truth_rev)
                metrics['SSDR'] = evaluation.evaluate_ssdr(pred_rev, truth_rev)
                    
        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'sample_metrics.csv'), float_format='%.8f', index=False)
        visualizer.plot_map(input_rev, pred_rev, truth_rev, timestamp, self.args.output_path, 'sample')
        print('Predict done.')

        return pred_rev
    
    def nowcast(self, model, input_loader):
        self.model = model
        self.model.to(self.args.device)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)['model'])
        self.model.eval()
        with torch.no_grad():
            for tensor, _ in input_loader:
                input_ = tensor.transpose(1, 0).to(self.args.device)
                input_ = scaler.minmax_norm(input_, self.args.vmax, self.args.vmin)
                preds = [self.model(input_) for _ in range(self.args.ensemble_members)]
                pred = torch.mean(torch.stack(preds), dim=0)
                pred = scaler.reverse_minmax_norm(pred, self.args.vmax, self.args.vmin)
        
        return pred

    def save_checkpoint(self, filename='checkpoint.pt'):
        states = {
            'iteration': self.current_iterations,
            'train_loss_g': self.train_loss_g,
            'train_loss_d': self.train_loss_d,
            'val_loss_g': self.val_loss_g,
            'val_loss_d': self.val_loss_d,
            'model': self.model.generator.state_dict(),
            'discriminator': self.model.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict()
        }
        torch.save(states, os.path.join(self.args.output_path, filename))

    def load_checkpoint(self, filename='checkpoint.pt'):
        states = torch.load(os.path.join(self.args.output_path, filename), map_location=self.args.device)
        return states
