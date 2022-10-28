import os
import torch
import numpy as np
import pandas as pd

import utils.visualizer as visualizer
import utils.evaluation as evaluation
import utils.scaler as scaler
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
        metrics['SSIM'] = []
        metrics['PSNR'] = []
        
        print('\n[Test]')
        for i, (t, elev, ref) in enumerate(test_loader):
            ref = scaler.minmax_norm(ref, self.args.vmax, self.args.vmin)
            masked_ref, mask, anchor, blockage_len = maskutils.gen_random_blockage_mask(
                ref, self.args.azimuth_blockage_range, self.args.random_seed + i)
            
            # forward
            output = maskutils.direct_filling(ref, self.args.azimuth_range[0], anchor, blockage_len)

            # back scaling
            ref = scaler.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
            output = scaler.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

            if (i + 1) % self.args.display_interval == 0:
                print('Batch: [{}][{}]'.format(i + 1, len(test_loader)))

            # evaluation
            metrics['MAE'].append(evaluation.evaluate_mae(ref[:, :1], output))
            metrics['RMSE'].append(evaluation.evaluate_rmse(ref[:, :1], output))
            metrics['COSSIM'].append(evaluation.evaluate_cossim(ref[:, :1], output))
            metrics['SSIM'].append(evaluation.evaluate_ssim(ref[:, :1], output))
            metrics['PSNR'].append(evaluation.evaluate_psnr(ref[:, :1], output))

        metrics['MAE'].append(np.mean(metrics['MAE'], axis=0))
        metrics['RMSE'].append(np.mean(metrics['RMSE'], axis=0))
        metrics['COSSIM'].append(np.mean(metrics['COSSIM'], axis=0))
        metrics['SSIM'].append(np.mean(metrics['SSIM'], axis=0))
        metrics['PSNR'].append(np.mean(metrics['PSNR'], axis=0))

        df = pd.DataFrame(data=metrics)
        df.to_csv(os.path.join(self.args.output_path, 'test_metrics.csv'), float_format='%.8f', index=False)
        tensors = torch.cat([output, ref], dim=1)
        visualizer.plot_ref(tensors, t, self.args.azimuth_range[0], self.args.radial_range[0],
                            anchor, blockage_len, self.args.output_path, 'test')
        print('Test done.')
    
    @torch.no_grad()
    def predict(self, sample_loader):
        metrics = {}
        metrics['MAE'] = []
        metrics['RMSE'] = []
        metrics['COSSIM'] = []
        metrics['SSIM'] = []
        metrics['PSNR'] = []

        print('\n[Predict]')
        for t, elev, ref in sample_loader:
            ref = scaler.minmax_norm(ref, self.args.vmax, self.args.vmin)
            masked_ref, mask, anchor, blockage_len = maskutils.gen_fixed_blockage_mask(
                ref, self.args.azimuth_range[0], self.args.sample_anchor, self.args.sample_blockage_len)
            
            # forward
            output = maskutils.direct_filling(ref, self.args.azimuth_range[0], anchor, blockage_len)

            # back scaling
            ref = scaler.reverse_minmax_norm(ref, self.args.vmax, self.args.vmin)
            output = scaler.reverse_minmax_norm(output, self.args.vmax, self.args.vmin)

            # evaluation
            metrics['MAE'].append(evaluation.evaluate_mae(ref[:, :1], output))
            metrics['RMSE'].append(evaluation.evaluate_rmse(ref[:, :1], output))
            metrics['COSSIM'].append(evaluation.evaluate_cossim(ref[:, :1], output))
            metrics['SSIM'].append(evaluation.evaluate_ssim(ref[:, :1], output))
            metrics['PSNR'].append(evaluation.evaluate_psnr(ref[:, :1], output))

        metrics['MAE'] = np.mean(metrics['MAE'], axis=0)
        metrics['RMSE'] = np.mean(metrics['RMSE'], axis=0)
        metrics['COSSIM'] = np.mean(metrics['COSSIM'], axis=0)
        metrics['SSIM'] = np.mean(metrics['SSIM'], axis=0)
        metrics['PSNR'] = np.mean(metrics['PSNR'], axis=0)

        df = pd.DataFrame(data=metrics, index=['MAE'])
        df.to_csv(os.path.join(self.args.output_path, 'predict_metrics.csv'), float_format='%.8f', index=False)
        tensors = torch.cat([output, ref], dim=1)
        visualizer.plot_ref(tensors, t, self.args.azimuth_range[0], self.args.radial_range[0],
                            anchor, blockage_len, self.args.output_path, 'predict')
        print('Predict done.')
