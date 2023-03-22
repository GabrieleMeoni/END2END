import os
import sys
sys.path.insert(1, os.path.join("..", "MSMatch"))
from utils import net_builder

try:
    from  onboard_prototype_utils import coarse_coregistration
except:
    raise ValueError("Impossible to find ""onboard_prototype_utils"".")

import torch
import numpy as np
from torch.nn.functional import pad

class oboardDetectorProcessor(torch.nn.Module):
    def __init__(self,  satellite, detector_number, model_name="efficientnet-lite0", num_classes=2, in_channels=3, depth=28, widen_factor=2, leaky_slope=0.1, dropout=0.0, patch_overlap=0):
        super().__init__()
        self.satellite=satellite
        self.detector_number=detector_number
        _net_builder=net_builder(
            model_name,
            False,
            {
                "depth": depth,
                "widen_factor": widen_factor,
                "leaky_slope": leaky_slope,
                "dropRate": dropout,
            },
        )

        self.ai_model= _net_builder(num_classes=num_classes, in_channels=in_channels)
        self.x_prev=None
        self.patch_overlap=patch_overlap

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        load_model = (
            checkpoint["eval_model"]
        )   
        self.ai_model.load_state_dict(load_model)


    def forward(self, x):
        x_coreg=coarse_coregistration(x, self.satellite, self.detector_number,  X_prev=self.x_prev, crop_empty_pixels=True)
        self.x_prev=x
        
        n_v, n_h=int(x_coreg.shape[1] / 256), int(x_coreg.shape[2] / 256)
        x_coreg_pad=pad(x_coreg, (0, (256 - (x_coreg.shape[2] - n_h * 256)) % 256, 0, (256 - (x_coreg.shape[1] - n_v * 256) % 256)), "replicate")
        y_demosaicked=torch.zeros([int(np.ceil(x_coreg_pad.shape[1] / (256-  self.patch_overlap))), int(np.ceil(x_coreg_pad.shape[2] / (256- self.patch_overlap)))])
        for n_h, h in enumerate(range(0, x_coreg_pad.shape[2], 256 - self.patch_overlap)):
            for n_v, v in enumerate(range(0, x_coreg_pad.shape[1], 256 - self.patch_overlap)):
                x_in=x_coreg_pad[:,v:v+256,h:h+256]
                y_demosaicked[n_v, n_h]= self.ai_model(x_in.unsqueeze(0)).max(1)[1]
        return y_demosaicked, x_coreg