import os
import torch
import numpy as np
import pandas as pd

import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.transform as transform
import utils.maskutils as maskutils


class BaselineTester:
    def __init__(self, args):
        self.args = args

    @torch.no_grad()
    def test(self, test_loader):
        metrics = {}
        metrics['MAE'] = []
        metrics['RMSE'] = []
        metrics['COSSIM'] = []
        
        print('\n[Test]')
        for i, (t, elev, ref) in enumerate(test_loader):
            ref = transform.minmax_norm(ref, self.args.vmax, self.args.vmin)
            masked_ref, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
                ref, self.args.azimuth_blockage_range, self.args.random_seed + i)
            
            # forward
            output = maskutils.direct_filling(ref, self.args.azimuth_range[0], anchor, blockage_len)

            # back scaling
            ref = transform.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
            output = transform.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

            if (i + 1) % self.args.display_interval == 0:
                print('Batch: [{}][{}]'.format(i + 1, len(test_loader)))

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
    def predict(self, sample_loader):
        print('\n[Predict]')
        for i, (t, elev, ref) in enumerate(sample_loader):
            metrics = {}
            ref = transform.minmax_norm(ref, self.args.vmax, self.args.vmin)
            masked_ref, mask, anchor, blockage_len = maskutils.gen_fixed_blockage_mask(
                ref, self.args.azimuth_range[0], self.args.sample_anchor, self.args.sample_blockage_len)
            
            # forward
            output = maskutils.direct_filling(ref, self.args.azimuth_range[0], anchor, blockage_len)

            # back scaling
            ref = transform.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
            output = transform.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

            # evaluation
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
        
