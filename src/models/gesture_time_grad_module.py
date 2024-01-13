from typing import Any, List

import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from gluonts.torch.util import copy_parameters
from src import utils
import hydra
import omegaconf
import pyrootutils
import ipdb
from scipy import linalg
from src.utils import metric


log = utils.get_pylogger(__name__)


class GestureTimeGradLightingModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            train_net: torch.nn.Module,
            prediction_net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True, ignore=["train_net"])
        self.save_hyperparameters(logger=True, ignore=["prediction_net"])

        self.train_net = train_net
        self.prediction_net = prediction_net
        self.train_step_count = 1
        self.alignmenter = metric.alignment(0.3, 2)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor, word: torch.Tensor, id: torch.Tensor, emo: torch.Tensor):
        trainer = self.trainer
        return self.train_net(trainer, x, cond, word, id, emo)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn'timestep store accuracy from these checks
        # self.val_acc_best.reset()
        pass

    def train_step(self, batch: Any):
        x = batch["x"]  # the output of body pose corresponding to the condition [80,95,45]
        log.info(f"x.shape:{x.shape}")
        cond = batch["cond"]  # [80,95,927]
        word = batch["word"]  # 
        id = batch["id"]  # 
        emo = batch["emo"]  # emo
        likelihoods, mean_loss = self.forward(x, cond, word, id, emo)
        return likelihoods, mean_loss

    def training_step(self, batch: Any, batch_idx: int):
        log.info(f"-------------------training_step----------------")
        likelihoods, loss = self.train_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        log.info(f"train_step count: {self.train_step_count}  train mean_loss: {loss}")
        self.train_step_count += 1
        return loss
        # return {"likelihoods": likelihoods, "mean_loss": mean_loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # self.train_acc.reset()
        self.train_step_count = 0

    def validation_step(self, batch: Any, batch_idx: int):
        log.info(f"-------------------validation_step----------------")
        likelihoods, loss = self.train_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        log.info(f"val mean_loss: {loss}")
        return {"loss": loss, "likelihoods": likelihoods}

    def   validation_epoch_end(self, outputs: List[Any]):
        pass

    def on_test_start(self):
        log.info('-----------------on_test_start--------------')
        copy_parameters(self.train_net, self.prediction_net)
        self.srgr_list = []
        self.fgd_list=[]
        self.align_list=[]
        log.info('-----------------copy_parameters--------------')
        # for name, train_net_para in self.train_net.named_parameters():
        #     log.info(f'train_net_para: {name}:\n {train_net_para.data}')
        # log.info('\n')
        # for name, prediction_net_para in self.prediction_net.named_parameters():
        #     # log.info(f'prediction_net_para: {name}:\n {prediction_net_para.data}')
        #     log.info(f'prediction_net_para: {name}:\n {prediction_net_para.data}')

    # !!!!!!!!!!!!!!!!!!!!!要修改吧
    def test_step(self, batch: Any, batch_idx: int):
        log.info("test_step----------------------------------------")
        x = batch["x"].cuda()  # the output of body pose corresponding to the condition [80,95,45]
        cond = batch["cond"].cuda()  # [1, 960000]  # [80,95,927]
        word = batch["word"].cuda()  # [1, 900]
        id = batch["id"].cuda()  # [1, 1]
        emo = batch["emo"].cuda()  # [1, 900] # emo
        sem = batch["sem"].cuda() 
        log.info(f"batch_idx:{batch_idx} ----------------------------------------")
        log.info(f"x.shape:{x.shape} cond.shape:{cond.shape}----------------------------------------")
        trainer = self.trainer
        output = self.prediction_net.forward(x, cond, word, id, emo, batch_idx, trainer)

        # print output.shape to log
        # log.info(f"output.shape:{output.shape}----------------------------------------")
        real_feats = x.squeeze(0)  
        generated_feats = output

        real_feats = real_feats.cpu().numpy()
        # generated_feats = generated_feats.cpu().numpy()
        
        # log.info(f"real_feats.shape:{real_feats.shape}----------------------------------------")
        # log.info(f"generated_feats.shape:{generated_feats.shape}----------------------------------------")
        sem_h = sem.cpu().numpy()
        # pose_dimes = real_feats.shape[1]
        pose_dimes = 47
        frame_num = real_feats.shape[0]
        threshold = 4
        #fgd
        fgd = self.calculate_fgd(real_feats, generated_feats)
        self.fgd_list.append(fgd)

        # Perform reverse standardization on generated_feats and real_feats

        # log.info(f"generated_feats.shape:{generated_feats.shape}----------------------------------------")
        # log.info(f"real_feats.shape:{real_feats.shape}----------------------------------------")
        # (855, 141)  , which means (length, 141)
        self.mean_pose = np.load("/home/lingling/code/DiffmotionEmotionGesture_v1/data/beat_cache/beat_4english_15_141_v3/train/bvh_rot/bvh_mean.npy")
        self.std_pose = np.load("/home/lingling/code/DiffmotionEmotionGesture_v1/data/beat_cache/beat_4english_15_141_v3/train/bvh_rot/bvh_std.npy")

        out_final = (generated_feats * self.std_pose) + self.mean_pose
        np_cat_results = out_final
        np_cat_targets = (real_feats * self.std_pose) + self.mean_pose

        #SRGR
        SRGR = self.calculate_srgr(np_cat_results, np_cat_targets, sem_h, pose_dimes, threshold)        
        self.srgr_list.append((SRGR, frame_num))
        
        #beat align
        t_start = 10
        t_end = 500
        in_audio = cond
        pose_fps = 15
        onset_raw, onset_bt, onset_bt_rms = self.alignmenter.load_audio(in_audio.cpu().numpy().reshape(-1), t_start, t_end, True)
        beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.alignmenter.load_pose(out_final, t_start, t_end, pose_fps, True)
        align = self.alignmenter.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, pose_fps)
        self.align_list.append(align)
        # x_cropped = x[:, -95:, :]
        # x_cropped_2d = x_cropped.flatten()
        # output_2d = output.reshape(-1, 45) 
        # x_cropped_2d = x_cropped.reshape(-1, 45)

        # log.info(f"x_cropped.shape:{x_cropped.shape}")
        # log.info(f"output.shape:{output.shape}")
        # log.info(f"x_cropped_2d.shape:{x_cropped_2d.shape}")
        # log.info(f"output_2d.shape:{output_2d.shape}")
        # fgd = self.calculate_fgd(x_cropped_2d, output_2d)  
        self.log("test/fgd", fgd, on_step=False, on_epoch=True, prog_bar=True)
        log.info(f"test fgd: {fgd}")

        self.log("test/SRGR", SRGR, on_step=False, on_epoch=True, prog_bar=True)
        log.info(f"test SRGR: {SRGR}")

        self.log("test/Beat Align", align, on_step=False, on_epoch=True, prog_bar=True)
        log.info(f"test Beat Align: {align}")
        # # ipdb.set_trace()
        # autoreg_all = batch["autoreg"].cuda()  # [20, 400, 45]
        # # log.info(f'test_step -> autoreg_all shape: {autoreg_all.shape} \n {autoreg_all}')
        # control_all = batch["control"].cuda()  # [80,400,27]
        # trainer = self.trainer
        # output = self.prediction_net.forward(autoreg_all, control_all, trainer)
        
        # x = batch["x"]  # the output of body pose corresponding to the condition [80,95,45]
        # cond = batch["cond"]  # [80,95,927]
        # word = batch["word"]  # 
        # id = batch["id"]  # 
        # emo = batch["emo"]  # emo
        # trainer = self.trainer
        # output = self.prediction_net.forward(x, cond, word, id, emo, trainer)
        return output

    def test_epoch_end(self, outputs: List[Any]):
        self.srgr_list = np.array(self.srgr_list)
        srgr_mean = self.calculate_weighted_average(self.srgr_list)
        log.info(f"srgr_mean: {srgr_mean}")
        
        self.fgd_list =np.array(self.fgd_list)
        fgd_mean = np.mean(self.fgd_list)
        log.info(f"test fgd mean: {fgd_mean}")
        
        self.align_list =np.array(self.align_list)
        align_mean = np.mean(self.align_list)
        log.info(f"test Beat Align mean: {align_mean}")
        # self.test_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }
    
    def calculate_weighted_average(self, value_list):
        total_weighted_sum = 0.0
        total_frame_num = 0

        for value, frame_num in value_list:
            weighted_value = value * frame_num
            total_weighted_sum += weighted_value
            total_frame_num += frame_num

        if total_frame_num == 0:
            return None 

        weighted_average = total_weighted_sum / total_frame_num
        return weighted_average

    def calculate_srgr(self, results, targets, semantic, pose_dimes, threshold):
        results = results.reshape(-1, pose_dimes, 3)
        targets = targets.reshape(-1, pose_dimes, 3)
        semantic = semantic.reshape(-1)
        
        min_samples = min(results.shape[0], targets.shape[0])
        results = results[:min_samples]
        targets = targets[:min_samples]
        
        diff = np.sum(abs(results-targets),2)
        success = np.where(diff < threshold, 1.0, 0.0)
        # success = np.ones_like(success)
        for i in range(success.shape[0]):
            # srgr == 0.165 when all success, scale range to [0, 1] 
            success[i, :] *= semantic[i] * (1/0.165) 
        rate = np.sum(success)/(success.shape[0]*success.shape[1])
        
        return rate   # The final calculated SRGR value should be weighted based on success.shape[0]


         
    def calculate_fgd(self, real_gestures, generated_gestures):
        A_mu = np.mean(real_gestures, axis=0)
        A_sigma = np.cov(real_gestures, rowvar=False)
        B_mu = np.mean(generated_gestures, axis=0)
        B_sigma = np.cov(generated_gestures, rowvar=False)
        try:
            frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
        except ValueError:
            frechet_dist = 1e+10
        return frechet_dist

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gesture_diffusion_lightningmodule.yaml")
    _ = hydra.utils.instantiate(cfg)
