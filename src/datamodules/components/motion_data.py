import numpy as np
from torch.utils.data import Dataset
from src import utils

log = utils.get_pylogger(__name__)


class MotionDataset(Dataset):
    """
    Motion dataset. 
    Prepares conditioning information (previous poses + control signal) and the corresponding next poses"""

    def __init__(self, control_data, joint_data, framerate, seqlen, n_lookahead, dropout):
        """
        Args:
        control_data: The control input
        joint_data: body pose input 
        Both with shape (samples, time-slices, features)
        seqlen: number of autoregressive body poses and previous control values
        n_lookahead: number of future control-values
        dropout: (0-1) dropout probability for previous poses
        """
        self.framerate = framerate
        self.seqlen = seqlen
        self.dropout = dropout
        seqlen_control = seqlen + n_lookahead + 1  # 1 is the current frame

        # For LSTM network
        n_frames = joint_data.shape[1]

        # Joint positions for n previous frames
        autoreg = self.concat_sequence(self.seqlen, joint_data[:, :n_frames - n_lookahead - 1,
                                                    :])  # [8428,95,225]  [B,S,F] F with previous gessture*5

        # Control for n previous frames + current frame audio
        control = self.concat_sequence(seqlen_control,
                                       control_data)  # [8428,95,702] [B,S,F] F with previous audio feature*n

        # conditioning

        # log.info("autoreg:" + str(autoreg.shape))
        # log.info("control:" + str(control.shape))
        new_cond = np.concatenate((autoreg, control),
                                  axis=2)  # [8428,95,927] [B,S,F`] F` with previous gesture features and audio features

        # joint positions for the current frame
        x_start = seqlen
        new_x = self.concat_sequence(1, joint_data[:, x_start:n_frames - n_lookahead, :])  # Output [8428,95,45]
        self.x = new_x  # Output [8428,95,45]
        self.cond = new_cond  # Input [8428,95,927] (n previous frames of gesture and audio feature)

        # TODO TEMP swap C and T axis to match existing implementation
        # self.x = np.swapaxes(self.x, 1, 2)
        # self.cond = np.swapaxes(self.cond, 1, 2)

        # log.info("self.x for predicting:" + str(self.x.shape))
        # log.info("self.cond:" + str(self.cond.shape))

    def n_channels(self):
        return self.x.shape[1], self.cond.shape[1]

    def concat_sequence(self, seqlen, data):
        """ 
        Concatenates a sequence of features to one.
        """
        nn, n_timesteps, n_feats = data.shape
        L = n_timesteps - (seqlen - 1)
        inds = np.zeros((L, seqlen)).astype(int)

        # create indices for the sequences we want
        rng = np.arange(0, n_timesteps)
        for ii in range(0, seqlen):
            inds[:, ii] = np.transpose(rng[ii:(n_timesteps - (seqlen - ii - 1))])

            # slice each sample into L sequences and store as new samples
        cc = data[:, inds, :].copy()

        # print ("cc: " + str(cc.shape))

        # reshape all timesteps and features into one dimention per sample
        dd = cc.reshape((nn, L, seqlen * n_feats))
        # print ("dd: " + str(dd.shape))
        return dd

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Returns poses and conditioning.
        If data-dropout sould be applied, a random selection of the previous poses is masked.
        The control is not masked
        """

        if self.dropout > 0.:
            self.x = np.swapaxes(self.x, 1, 2)
            self.cond = np.swapaxes(self.cond, 1, 2)

            n_feats, tt = self.x[idx, :, :].shape
            cond_masked = self.cond[idx, :, :].copy()

            keep_pose = np.random.rand(self.seqlen, tt) < (1 - self.dropout)

            # print(keep_pose)
            n_cond = cond_masked.shape[0] - (n_feats * self.seqlen)
            mask_cond = np.full((n_cond, tt), True)

            mask = np.repeat(keep_pose, n_feats, axis=0)
            mask = np.concatenate((mask, mask_cond), axis=0)
            # print(mask)

            cond_masked = cond_masked * mask
            self.x = np.swapaxes(self.x, 1, 2)
            self.cond = np.swapaxes(self.cond, 1, 2)
            cond_masked = np.swapaxes(cond_masked, 0, 1)
            sample = {'x': self.x[idx, :, :], 'cond': cond_masked}
        else:
            sample = {'x': self.x[idx, :, :], 'cond': self.cond[idx, :, :]}

        return sample


class TestDataset(Dataset):
    """Test dataset."""

    def __init__(self, control_data, joint_data):
        """
        Args:
        control_data: The control input
        joint_data: body pose input 
        Both with shape (samples, time-slices, features)
        """
        # Joint positions
        self.autoreg = joint_data

        # Control
        self.control = control_data

    def __len__(self):
        return self.autoreg.shape[0]

    def __getitem__(self, idx):
        sample = {'autoreg': self.autoreg[idx, :], 'control': self.control[idx, :]}
        return sample
