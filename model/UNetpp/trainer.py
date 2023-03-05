import os
import math
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from utils.earlystopping import EarlyStopping
import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.transform as transform
import utils.maskutils as maskutils
import utils.losses as losses


class UNetpp_GAN_Trainer:
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
        
        self.optimizer_g = Adam(self.model.generator.parameters(), lr=self.args.learning_rate,
                                betas=(self.args.beta1, self.args.beta2),
                                weight_decay=self.args.weight_decay)
        self.optimizer_d = Adam(self.model.discriminator.parameters(), lr=self.args.learning_rate,
                                betas=(self.args.beta1, self.args.beta2),
                                weight_decay=self.args.weight_decay)

        if self.args.train:
            self.train()
        if self.args.test:
            self.test()
    
    def train(self):
        self.count_params()
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
            start_epoch = int(math.floor(self.current_iterations / len(self.train_loader)))
        else:
            self.current_iterations = 0
            self.train_loss_g = []
            self.train_loss_d = []
            self.val_loss_g = []
            self.val_loss_d = []
            start_epoch = 0
        
        early_stopping = EarlyStopping(verbose=True, path='bestmodel.pth')

        for epoch in range(start_epoch, self.total_epochs):
            print('\n[Train]')
            print('Epoch: [{}][{}]'.format(epoch + 1, self.total_epochs))
            train_loss_g = []
            train_loss_d = []
            val_loss_g = []
            val_loss_d = []

            # Train
            self.model.train()
            for i, (t, elev, ref) in enumerate(self.train_loader):
                self.current_iterations += 1
                
                # load data
                ref = ref.to(self.args.device)
                ref = transform.minmax_norm(ref, self.args.vmax, self.args.vmin)
                masked_ref, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
                    ref, self.args.azimuth_blockage_range, self.args.seed + i)
                
                # discriminator forward
                input_g = torch.cat([masked_ref, mask], dim=1)
                output_g = self.model(input_g)
                output_g = masked_ref[:, :1] + output_g * (1 - mask[:, :1])
                real_input_g = torch.cat([ref[:, :1], mask[:, :1]], dim=1)
                fake_input_g = torch.cat([output_g.detach(), mask[:, :1]], dim=1)
                real_score = self.model.discriminator(real_input_g)
                fake_score = self.model.discriminator(fake_input_g)
                loss_d = losses.cal_d_loss(fake_score, real_score)
                
                # discriminator backward
                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

                # generator backward
                fake_input_g = torch.cat([output_g, mask[:, :1]], dim=1)
                fake_score = self.model.discriminator(fake_input_g)
                loss_g = losses.cal_g_loss(fake_score) + \
                    self.args.weight_recon * losses.biased_mae_loss(output_g, ref[:, :1], self.args.vmax, self.args.vmin)
                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()
                
                # back scaling
                ref = transform.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
                masked_ref = transform.reverse_minmax_norm(masked_ref, self.args.vmax, self.args.vmin)
                output_g = transform.reverse_minmax_norm(output_g, self.args.vmax, self.args.vmin)

                train_loss_g.append(loss_g.item())
                train_loss_d.append(loss_d.item())
                if (i + 1) % self.args.display_interval == 0:
                    print('Epoch: [{}][{}] Batch: [{}][{}] Loss G: {:.6f} Loss D: {:.6f}'.format(
                        epoch + 1, self.total_epochs, i + 1, len(self.train_loader), loss_g.item(), loss_d.item()))
                
                # Check max iterations
                if self.current_iterations >= self.args.max_iterations:
                    print('Max interations %d reached.' % self.args.max_iterations)
                    break
            
            self.train_loss_g.append(np.mean(train_loss_g))
            self.train_loss_d.append(np.mean(train_loss_d))
            print('Epoch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                epoch + 1, self.total_epochs, self.train_loss_g[-1], self.train_loss_d[-1]))
            
            np.savetxt(os.path.join(self.args.output_path, 'train_loss_g.txt'), self.train_loss_g)
            np.savetxt(os.path.join(self.args.output_path, 'train_loss_d.txt'), self.train_loss_d)

            # Save tensors
            tensors = torch.cat([output_g, ref], dim=1)
            visualizer.save_tensor(tensors, self.args.output_path, 'train')
            print('Tensors saved')
           
            # Validate
            print('\n[Val]')
            self.model.eval()
            with torch.no_grad():
                for i, (t, elev, ref) in enumerate(self.val_loader):
                    # load data
                    ref = ref.to(self.args.device)
                    ref = transform.minmax_norm(ref, self.args.vmax, self.args.vmin)
                    masked_ref, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
                        ref, self.args.azimuth_blockage_range, self.args.seed + i)
                    
                    # discriminator forward
                    input_g = torch.cat([masked_ref, mask], dim=1)
                    output_g = self.model(input_g)
                    output_g = masked_ref[:, :1] + output_g * (1 - mask[:, :1])
                    real_input_g = torch.cat([ref[:, :1], mask[:, :1]], dim=1)
                    fake_input_g = torch.cat([output_g.detach(), mask[:, :1]], dim=1)
                    real_score = self.model.discriminator(real_input_g)
                    fake_score = self.model.discriminator(fake_input_g)
                    loss_d = losses.cal_d_loss(fake_score, real_score)

                    # generator forward
                    fake_score = self.model.discriminator(fake_input_g)
                    loss_g = losses.cal_g_loss(fake_score) + \
                        self.args.weight_recon * losses.biased_mae_loss(output_g, ref[:, :1], self.args.vmax, self.args.vmin)

                    # back scaling
                    ref = transform.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
                    masked_ref = transform.reverse_minmax_norm(masked_ref, self.args.vmax, self.args.vmin)
                    output_g = transform.reverse_minmax_norm(output_g, self.args.vmax, self.args.vmin)

                    val_loss_g.append(loss_g.item())
                    val_loss_d.append(loss_d.item())
                    if (i + 1) % self.args.display_interval == 0:
                        print('Epoch: [{}][{}] Batch: [{}][{}] Loss G: {:.6f} Loss D: {:.6f}'.format(
                            epoch + 1, self.total_epochs, i + 1, len(self.val_loader), loss_g.item(), loss_d.item()))
                            
            self.val_loss_g.append(np.mean(val_loss_g))
            self.val_loss_d.append(np.mean(val_loss_d))
            print('Epoch: [{}][{}] Loss G: {:.4f} Loss D: {:.4f}'.format(
                epoch + 1, self.total_epochs, self.val_loss_g[-1], self.val_loss_d[-1]))

            np.savetxt(os.path.join(self.args.output_path, 'val_loss_g.txt'), self.val_loss_g)
            np.savetxt(os.path.join(self.args.output_path, 'val_loss_d.txt'), self.val_loss_d)

            # Save tensors
            tensors = torch.cat([output_g, ref], dim=1)
            visualizer.save_tensor(tensors, self.args.output_path, 'val')
            print('Tensors saved')

            # Plot loss
            visualizer.plot_loss(self.train_loss_g, self.val_loss_g, self.args.output_path, 'loss_g.png')
            visualizer.plot_loss(self.train_loss_d, self.val_loss_d, self.args.output_path, 'loss_d.png')

            # Save checkpoint
            self.save_checkpoint()

            if self.args.early_stopping:
                early_stopping(self.val_loss_g[-1], self)
            if early_stopping.early_stop:
                break

    @torch.no_grad()
    def test(self):
        metrics = {}
        metrics['MAE'] = []
        metrics['RMSE'] = []
        metrics['COSSIM'] = []
        
        print('\n[Test]')
        self.model.generator.load_state_dict(self.load_checkpoint('bestmodel.pth')['model'])
        self.model.discriminator.load_state_dict(self.load_checkpoint('bestmodel.pth')['discriminator'])
        self.model.eval()
        
        for i, (t, elev, ref) in enumerate(self.test_loader):
            ref = ref.to(self.args.device)
            ref = transform.minmax_norm(ref, self.args.vmax, self.args.vmin)
            masked_ref, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
                ref, self.args.azimuth_blockage_range, self.args.seed + i)
            
            # discriminator forward
            input_g = torch.cat([masked_ref, mask], dim=1)
            output_g = self.model(input_g)
            output_g = masked_ref[:, :1] + output_g * (1 - mask[:, :1])

            # back scaling
            ref = transform.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
            masked_ref = transform.reverse_minmax_norm(masked_ref, self.args.vmax, self.args.vmin)
            output_g = transform.reverse_minmax_norm(output_g, self.args.vmax, self.args.vmin)

            if (i + 1) % self.args.display_interval == 0:
                print('Batch: [{}][{}]'.format(i + 1, len(self.test_loader)))

            # evaluation
            metrics['MAE'].append(evaluation.evaluate_mae(ref[:, :1], output_g, mask[:, :1]))
            metrics['RMSE'].append(evaluation.evaluate_rmse(ref[:, :1], output_g, mask[:, :1]))
            metrics['MBE'].append(evaluation.evaluate_mbe(ref[:, :1], output_g, mask[:, :1]))
            metrics['COSSIM'].append(evaluation.evaluate_cossim(ref[:, :1], output_g, mask[:, :1]))

        metrics['MAE'].append(np.mean(metrics['MAE'], axis=0))
        metrics['RMSE'].append(np.mean(metrics['RMSE'], axis=0))
        metrics['MBE'].append(np.mean(metrics['MBE'], axis=0))
        metrics['COSSIM'].append(np.mean(metrics['COSSIM'], axis=0))
        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'test_metrics.csv'), float_format='%.8f', index=False)
        
        # Save tensors
        tensors = torch.cat([output_g, ref], dim=1)
        visualizer.save_tensor(tensors, self.args.output_path, 'test')
        print('Tensors saved')
        
        # Plot tensors
        print('\nVisualizing...')
        visualizer.plot_ppi(tensors, t, self.args.azimuth_range[0], self.args.radial_range[0],
                            anchor, blockage_len, self.args.output_path, 'test')
        print('Visualization complete')
        print('\nTest complete')

    @torch.no_grad()
    def predict(self, model, sample_loader):
        print('\n[Predict]')
        model.generator.load_state_dict(self.load_checkpoint('bestmodel.pth')['model'])
        model.discriminator.load_state_dict(self.load_checkpoint('bestmodel.pth')['discriminator'])
        model.eval()
        
        for i, (t, elev, ref) in enumerate(sample_loader):
            print('\nSample {}'.format(i))
            metrics = {}
            ref = ref.to(self.args.device)
            ref = transform.minmax_norm(ref, self.args.vmax, self.args.vmin)
            if isinstance(self.args.sample_anchor, int):
                self.args.sample_anchor = [self.args.sample_anchor]
            if isinstance(self.args.sample_blockage_len, int):
                self.args.sample_blockage_len = [self.args.sample_blockage_len]
            masked_ref, mask, anchor, blockage_len = maskutils.gen_fixed_blockage_mask(
                ref, self.args.azimuth_range[0], self.args.sample_anchor[i], self.args.sample_blockage_len[i])
            
            # discriminator forward
            input_g = torch.cat([masked_ref, mask], dim=1)
            output_g = self.model(input_g)
            output_g = masked_ref[:, :1] + output_g * (1 - mask[:, :1])

            # back scaling
            ref = transform.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
            masked_ref = transform.reverse_minmax_norm(masked_ref, self.args.vmax, self.args.vmin)
            output_g = transform.reverse_minmax_norm(output_g, self.args.vmax, self.args.vmin)

            # evaluation
            print('\nEvaluating...')
            metrics['MAE'] = evaluation.evaluate_mae(ref[:, :1], output_g, mask[:, :1])
            metrics['RMSE'] = evaluation.evaluate_rmse(ref[:, :1], output_g, mask[:, :1])
            metrics['MBE'] = evaluation.evaluate_mbe(ref[:, :1], output_g, mask[:, :1])
            metrics['COSSIM'] = evaluation.evaluate_cossim(ref[:, :1], output_g, mask[:, :1])

            df = pd.DataFrame(data=metrics, index=['MAE'])
            df.to_csv(os.path.join(self.args.output_path, 'sample_{}_metrics.csv'.format(i)), float_format='%.8f', index=False)
            print('Evaluation complete')

            # Save tensors
            tensors = torch.cat([output_g, ref], dim=1)
            visualizer.save_tensor(tensors, self.args.output_path, 'sample_{}'.format(i))
            print('Tensors saved')
            
            # Plot tensors
            print('\nVisualizing...')
            visualizer.plot_ppi(tensors, t, self.args.azimuth_range[0], self.args.radial_range[0],
                                anchor, blockage_len, self.args.output_path, 'sample_{}'.format(i))
            visualizer.plot_psd(tensors, self.args.radial_range[0], anchor, blockage_len,
                                self.args.output_path, 'sample_{}'.format(i))
            print('Visualization complete')

        print('\nPrediction complete')

    def save_checkpoint(self, filename='checkpoint.pth'):
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

    def load_checkpoint(self, filename='checkpoint.pth'):
        states = torch.load(os.path.join(self.args.output_path, filename), map_location=self.args.device)
        return states

    def count_params(self):
        G_params = filter(lambda p: p.requires_grad, self.model.generator.parameters())
        D_params = filter(lambda p: p.requires_grad, self.model.discriminator.parameters())
        num_G_params = sum([p.numel() for p in G_params])
        num_D_params = sum([p.numel() for p in D_params])
        print('\nModel name: {}'.format(type(self.model).__name__))
        print('G params: {}'.format(num_G_params))
        print('D params: {}'.format(num_D_params))
        print('Total params: {}'.format(num_G_params + num_D_params))
