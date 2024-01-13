from typing import Optional
from pytorch_lightning import LightningDataModule
import joblib as jl
import os
from src import utils
import numpy as np
from src.datamodules.components.motion_data import MotionDataset, TestDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import Logger
# LightningLoggerBase
from src import utils
from src.pymo.writers import *
import hydra
from torch.utils.data import Dataset
import torch
from scipy import stats
import pandas as pd
import glob
log = utils.get_pylogger(__name__)

class GestureDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = "data/GesturesData",
                 framerate: str = 20,
                 seqlen: int = 5,
                 n_lookahead: int = 20,
                 dropout: float = 0.4,
                 batch_size: int = 80,
                 input_size: int = 972,
                 num_workers: int = 0 ,#16
                 cache_name: str = 'cache'
                 # test: int = 0,
        ):
        super(GestureDataModule, self).__init__()
        # region variable define
        self.test_output = None
        self.val_output = None
        self.val_input = None
        self.train_output = None
        self.train_input = None
        log.info(f"-------------------Init GestureDataModule----------------")
        self.test_dataset = None
        self.n_test = None
        self.n_cond_channels = None
        self.n_x_channels = None

        self.val_data_loader = None
        self.train_data_loader = None
        self.test_input = None
        self.validation_dataset = None
        self.data_pipe = None
        self.output_scaler = None
        self.input_scaler = None
        self.train_dataset = None

        self.data_root = data_dir
        self.framerate = framerate
        self.seqlen = seqlen
        self.n_lookahead = n_lookahead
        self.dropout = dropout
        self.batch_size = batch_size
        self.feature_length = None
        self.num_workers = num_workers
        self.cache_name = cache_name
        # endregion
        # this line allows to access init params with 'self.hparams' attribute
        
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        log.info('-----------------prepare_data 啥都不做_______________________')
        # log.info(f"Diff Flow Datamodule => {sys._getframe().f_code.co_name}()")
        #load scalers
        # print(os.path.join(self.data_root, 'input_scaler.sav'))
        # self.input_scaler = jl.load(os.path.join(self.data_root, 'input_scaler.sav'))
        # self.output_scaler = jl.load(os.path.join(self.data_root, 'output_scaler.sav'))
        # log.info(self.input_scaler)
        # log.info(self.output_scaler)

        # load pipeline for conversion from motion features to BVH.
        # self.data_pipe = jl.load(os.path.join(self.data_root, 'data_pipe_' + str(self.framerate) + 'fps.sav'))
        # # log.info(self.data_pipe)
        # # load the data. This should already be Standardized
        # self.train_input = np.load(os.path.join(self.data_root, 'train_input_' + str(self.framerate) + 'fps.npz'))[
        #     'clips'].astype(np.float32)  # Train Audio input [8428,120,27] [B,S,F]
        # self.train_output = np.load(os.path.join(self.data_root, 'train_output_' + str(self.framerate) + 'fps.npz'))[
        #     'clips'].astype(np.float32)  # Train Gesture output [8428,120,45] [B,S,F]

        # # log.info(f'self train_output mean: \n {mean_data}, \n {mean_data.shape}')
        # self.val_input = np.load(os.path.join(self.data_root, 'val_input_' + str(self.framerate) + 'fps.npz'))[
        #     'clips'].astype(np.float32)  # Value Audio input[264,120,27] [B,S,F]
        # self.val_output = np.load(os.path.join(self.data_root, 'val_output_' + str(self.framerate) + 'fps.npz'))[
        #     'clips'].astype(np.float32)  # Value Gesture output[264,120,45] [B,S,F]
        # # log.info(np.shape(train_input) + np.shape(train_output) + np.shape(val_input) + np.shape(val_output))
        # self.test_input = np.load(os.path.join(self.data_root, 'dev_input_' + str(self.framerate) + 'fps.npz'))[
        #     'clips'].astype(np.float32)
        # self.test_output = np.load(os.path.join(self.data_root, 'dev_output_' + str(self.framerate) + 'fps.npz'))[
        #     'clips'].astype(np.float32)
        # log.info(
        #     f'prepare_data --> test_input shape: {self.test_input.shape} test_output shape: {self.test_output.shape}')
        # log.info(self.test_input)

    def setup(self, stage: Optional[str] = None) -> None:
        log.info('-----------------setup_______________________')
        # 
        if stage in (None, "fit"):
            log.info(f'-----------------setup stage: {stage}')
            # Create pytorch data sets
            # # beat
            self.train_dataset = __import__(f"src.datamodules.components.beat", fromlist=["something"]).CustomDataset(loader_type = "train",cache_name = self.cache_name)
            self.validation_dataset = __import__(f"src.datamodules.components.beat", fromlist=["something"]).CustomDataset(loader_type = "val",cache_name = self.cache_name)
            # diffusion
            # self.train_dataset = MotionDataset(control_data=self.train_input, joint_data=self.train_output,
            #                                    framerate=self.framerate,
            #                                    seqlen=self.seqlen, n_lookahead=self.n_lookahead,
            #                                    dropout=self.dropout)
            # self.validation_dataset = MotionDataset(control_data=self.val_input, joint_data=self.val_output,
            #                                         framerate=self.framerate,
            #                                         seqlen=self.seqlen, n_lookahead=self.n_lookahead,
            #                                         dropout=self.dropout)
        if stage in (None, "test"):
            log.info(f'-----------------setup stage: {stage}')
            # # test data for network tuning. It contains the same data as val_input, but sliced into longer 20-sec
            # # exerpts
            # # make sure the test data is at least one batch size
            # self.n_test = self.test_input.shape[0]  # 20

            # # n_tiles = 1 + hparams.Train.batch_size // self.n_test
            # n_tiles = 1 + self.batch_size // self.n_test  # 5
            # test_input = np.tile(self.test_input.copy(), (n_tiles, 1, 1))  # [100,400,27]
            # # test_output = np.tile(self.test_output.copy(), (n_tiles, 1, 1))

            # # initialise test output with zeros (mean pose)
            # # n_x_channels: [45]  n_cond_channel: [927]
            # self.n_x_channels = self.output_scaler.mean_.shape[0]
            # self.n_cond_channels = self.n_x_channels * self.seqlen + test_input.shape[2] * (
            #         self.seqlen + 1 + self.n_lookahead)
            # self.feature_length = self.n_x_channels + self.n_cond_channels
            # test_output = np.zeros((test_input.shape[0], test_input.shape[1], self.n_x_channels)).astype(
            #     np.float32)  # [100,400,45]
            # # log.info(f'setup --> test_output shape: {test_output.shape} \n {test_output}')
            # sr:
            # self.test_dataset = TestDataset(test_input, test_output)
            # beat
            print("test_dataloader---------------------------------")
            self.test_dataset = __import__(f"src.datamodules.components.beat",
                                                 fromlist=["something"]).CustomDataset(loader_type = "test",cache_name = self.cache_name)
            print("test_dataloader  结束---------------------------------")

    def n_channels(self):
        return self.n_x_channels, self.n_cond_channels

    def train_dataloader(self):
        # train_data_loader = DataLoader(
        #     dataset=self.train_dataset,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=True,
        #     drop_last=True,
        # )

        #beat
        train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        # beat sr:
        # train_data_loader = DataLoader(
        #     self.train_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=False if self.ddp else True,
        #     num_workers=args.loader_workers,
        #     drop_last=True,
        #     sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if self.ddp else None,
        # )
        return train_data_loader

    def val_dataloader(self):
        # val_data_loader = DataLoader(
        #     dataset=self.validation_dataset,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False,
        #     drop_last=True
        # )
        # beat
        val_data_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,#false
        )
        #beat sr
        # val_data_loader = torch.utils.data.DataLoader(
        #     self.validation_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     num_workers=args.loader_workers,
        #     drop_last=False,
        #     sampler=torch.utils.data.distributed.DistributedSampler(self.val_data) if self.ddp else None,
        # )

        return val_data_loader

    def test_dataloader(self):
        log.info("test_Dataloader start……………………………………………………………………………………")
        test_data_loader = DataLoader(
            self.test_dataset,
            batch_size= 1, #self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True
        )
        log.info("test_Dataloader end……………………………………………………………………………………")
        return test_data_loader

    def predict_dataloader(self):
        predict_data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True
        )
        return predict_data_loader

    def inv_standardize(self, data, scaler):
        shape = data.shape
        flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
        scaled = scaler.inverse_transform(flat).reshape(shape)
        return scaled

    def save_animation(self, motion_data, filename):
        print('-----save animation-------------')
        # print(f'motion_data shape: {motion_data.shape}')
        # control_data = control_data.cpu().numpy()
        # motion_data = motion_data.cpu().numpy()
        anim_clips = self.inv_standardize(motion_data[:self.n_test, :, :], self.output_scaler)
        # print(f'anim_clips shape: {anim_clips.shape}')
        np.savez(filename + ".npz", clips=anim_clips)
        self.write_bvh(anim_clips, filename)

    def write_bvh(self, anim_clips, filename):
        print('inverse_transform...')
        inv_data = self.data_pipe.inverse_transform(anim_clips)
        writer = BVHWriter()
        for i in range(0, anim_clips.shape[0]):
            filename_ = f'{filename}_{str(i)}.bvh'
            print('writing:' + filename_)
            with open(filename_, 'w') as f:
                writer.write(inv_data[i], f, framerate=self.framerate)
    
    def result2target_vis(self, res_bvhlist, save_path, demo_name, verbose=True):
        from src.datamodules.env import joints_list
        ori_list = joints_list["beat_joints"]
        target_list = joints_list["spine_neck_141"]
        file_content_length = 431
        
        bvh_files_dirs = sorted(glob.glob(f'{res_bvhlist}/*.bvh'), key=str)
        counter = 0
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, bvh_file_dir in enumerate(bvh_files_dirs):
            short_name = bvh_file_dir.split("/")[-1][11:]
            #print(short_name)
            wirte_file =  open(os.path.join(save_path, f'res_{short_name}'),'w+')
            with open(f"{demo_name}{short_name}",'r') as pose_data_pre:
                pose_data_pre_file = pose_data_pre.readlines()
                for j, line in enumerate(pose_data_pre_file[0:file_content_length]):
                        wirte_file.write(line)
                offset_data = pose_data_pre_file[file_content_length]
                offset_data = np.fromstring(offset_data, dtype=float, sep=' ')
            wirte_file.close()

            wirte_file = open(os.path.join(save_path, f'res_{short_name}'),'r')
            ori_lines = wirte_file.readlines()
            with open(bvh_file_dir, 'r') as pose_data:
                pose_data_file = pose_data.readlines()
            ori_lines[file_content_length-2] = 'Frames: ' + str(len(pose_data_file)-1) + '\n'
            wirte_file.close() 

            wirte_file = open(os.path.join(save_path, f'res_{short_name}'),'w+')
            wirte_file.writelines(i for i in ori_lines[:file_content_length])    
            wirte_file.close() 

            with open(os.path.join(save_path, f'res_{short_name}'),'a+') as wirte_file: 
                with open(bvh_file_dir, 'r') as pose_data:
                    data_each_file = []
                    pose_data_file = pose_data.readlines()
                    for j, line in enumerate(pose_data_file):
                        if not j:
                            pass
                        else:          
                            data = np.fromstring(line, dtype=float, sep=' ')
                            data_rotation = offset_data.copy()   
                            for iii, (k, v) in enumerate(target_list.items()): # here is 147 rotations by 3
                                #print(data_rotation[ori_list[k][1]-v:ori_list[k][1]], data[iii*3:iii*3+3])
                                data_rotation[ori_list[k][1]-v:ori_list[k][1]] = data[iii*3:iii*3+3]
                            data_each_file.append(data_rotation)
            
                for line_data in data_each_file:
                    line_data = np.array2string(line_data, max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                    wirte_file.write(line_data[1:-2]+'\n')

            counter += 1
            if verbose:
                logger.info('data_shape:', data_rotation.shape, 'process:', counter, '/', len(bvh_files_dirs))
