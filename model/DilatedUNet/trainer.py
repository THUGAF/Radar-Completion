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


class DilatedUNet_Trainer:
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

        self.optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

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
            start_epoch = int(math.floor(self.current_iterations / len(self.train_loader)))
        else:
            self.current_iterations = 0
            self.train_loss = []
            self.val_loss = []
            start_epoch = 0
        
        early_stopping = EarlyStopping(verbose=True, path='bestmodel.pth')

        # Train
        for epoch in range(start_epoch, self.total_epochs):
            print('\n[Train]')
            print('Epoch: [{}][{}]'.format(epoch + 1, self.total_epochs))
            train_loss = []
            val_loss = []

            # Train
            self.model.train()

            for i, (t, elev, ref) in enumerate(self.train_loader):
                self.current_iterations += 1
                
                # load data
                ref = ref.to(self.args.device)
                ref = transform.minmax_norm(ref, self.args.vmax, self.args.vmin)
                masked_ref, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
                    ref, self.args.azimuth_blockage_range, self.args.seed + i)
                
                # forward
                input_ = torch.cat([masked_ref, mask], dim=1)
                output = self.model(input_)
                output = masked_ref[:, :1] + output * (1 - mask[:, :1])
                loss = self.args.weight_recon * losses.biased_mae_loss(output, ref[:, :1], self.args.vmax, self.args.vmin)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
   
                # back scaling
                ref = transform.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
                masked_ref = transform.reverse_minmax_norm(masked_ref, self.args.vmax, self.args.vmin)
                output = transform.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

                train_loss.append(loss.item())
                if (i + 1) % self.args.display_interval == 0:
                    print('Epoch: [{}][{}] Batch: [{}][{}] Loss: {:.6f}'.format(
                        epoch + 1, self.total_epochs, i + 1, len(self.train_loader), loss.item()))
                
                # Check max iterations
                if self.current_iterations >= self.args.max_iterations:
                    print('Max interations %d reached.' % self.args.max_iterations)
                    break
            
            # Save loss
            self.train_loss.append(np.mean(train_loss))
            np.savetxt(os.path.join(self.args.output_path, 'train_loss.txt'), self.train_loss)
            print('Epoch: [{}][{}] Loss: {:.6f}'.format(epoch + 1, self.total_epochs, self.train_loss[-1]))
            
            # Save tensors
            tensors = torch.cat([output, ref], dim=1)
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

                    # forward
                    input_ = torch.cat([masked_ref, mask], dim=1)
                    output = self.model(input_)
                    output = masked_ref[:, :1] + output * (1 - mask[:, :1])
                    loss = self.args.weight_recon * losses.biased_mae_loss(output, ref[:, :1], self.args.vmax, self.args.vmin)

                    # back scaling
                    ref = transform.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
                    masked_ref = transform.reverse_minmax_norm(masked_ref, self.args.vmax, self.args.vmin)
                    output = transform.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

                    val_loss.append(loss.item())
                    if (i + 1) % self.args.display_interval == 0:
                        print('Epoch: [{}][{}] Batch: [{}][{}] Loss: {:.6f}'.format(
                            epoch + 1, self.total_epochs, i + 1, len(self.val_loader), loss.item()))
            
            # Save loss                
            self.val_loss.append(np.mean(val_loss))         
            np.savetxt(os.path.join(self.args.output_path, 'val_loss.txt'), self.val_loss)
            print('Epoch: [{}][{}] Loss: {:.6f}'.format(epoch + 1, self.total_epochs, self.val_loss[-1]))
            
            # Save tensors
            tensors = torch.cat([output, ref], dim=1)
            visualizer.save_tensor(tensors, self.args.output_path, 'train')
            print('Tensors saved')
            
            # Plot loss
            visualizer.plot_loss(self.train_loss, self.val_loss, self.args.output_path, 'loss.png')

            # Save checkpoint
            self.save_checkpoint()

            # Check early stopping
            if self.args.early_stopping:
                early_stopping(self.val_loss[-1], self)
            if early_stopping.early_stop:
                break
    
    @torch.no_grad()
    def test(self):
        metrics = {}
        metrics['MAE'] = []
        metrics['RMSE'] = []
        metrics['COSSIM'] = []
        
        print('\n[Test]')
        self.model.load_state_dict(self.load_checkpoint('bestmodel.pth')['model'])
        self.model.eval()
        
        for i, (t, elev, ref) in enumerate(self.test_loader):
            ref = ref.to(self.args.device)
            ref = transform.minmax_norm(ref, self.args.vmax, self.args.vmin)
            masked_ref, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
                ref, self.args.azimuth_blockage_range, self.args.seed + i)
            
            # forward
            input_ = torch.cat([masked_ref, mask], dim=1)
            output = self.model(input_)
            output = masked_ref[:, :1] + output * (1 - mask[:, :1])

            # back scaling
            ref = transform.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
            masked_ref = transform.reverse_minmax_norm(masked_ref, self.args.vmax, self.args.vmin)
            output = transform.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

            if (i + 1) % self.args.display_interval == 0:
                print('Batch: [{}][{}]'.format(i + 1, len(self.test_loader)))

            # evaluation
            metrics['MAE'].append(evaluation.evaluate_mae(ref[:, :1], output, mask[:, :1]))
            metrics['RMSE'].append(evaluation.evaluate_rmse(ref[:, :1], output, mask[:, :1]))
            metrics['COSSIM'].append(evaluation.evaluate_cossim(ref[:, :1], output, mask[:, :1]))

        metrics['MAE'].append(np.mean(metrics['MAE'], axis=0))
        metrics['RMSE'].append(np.mean(metrics['RMSE'], axis=0))
        metrics['COSSIM'].append(np.mean(metrics['COSSIM'], axis=0))
        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'test_metrics.csv'), float_format='%.8f', index=False)
        
        # Save tensors
        tensors = torch.cat([output, ref], dim=1)
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
        model.load_state_dict(self.load_checkpoint('bestmodel.pth')['model'])
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
            
            # forward
            input_ = torch.cat([masked_ref, mask], dim=1)
            output = model(input_)
            output = masked_ref[:, :1] + output * (1 - mask[:, :1])

            # back scaling
            ref = transform.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
            masked_ref = transform.reverse_minmax_norm(masked_ref, self.args.vmax, self.args.vmin)
            output = transform.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

            # evaluation
            print('\nEvaluating...')
            metrics['MAE'] = evaluation.evaluate_mae(ref[:, :1], output, mask[:, :1])
            metrics['RMSE'] = evaluation.evaluate_rmse(ref[:, :1], output, mask[:, :1])
            metrics['COSSIM'] = evaluation.evaluate_cossim(ref[:, :1], output, mask[:, :1])

            df = pd.DataFrame(data=metrics, index=['MAE'])
            df.to_csv(os.path.join(self.args.output_path, 'sample_{}_metrics.csv'.format(i)), float_format='%.8f', index=False)
            print('Evaluation complete')

            # Save tensors
            tensors = torch.cat([output, ref], dim=1)
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
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(states, os.path.join(self.args.output_path, filename))

    def load_checkpoint(self, filename='checkpoint.pth'):
        states = torch.load(os.path.join(self.args.output_path, filename), map_location=self.args.device)
        return states
    
    def count_params(self):
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        num_params = sum([p.numel() for p in model_params])
        print('\nModel name: {}'.format(type(self.model).__name__))
        print('Total params: {}'.format(num_params))
