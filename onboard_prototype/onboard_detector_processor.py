import os
import sys

sys.path.insert(1, os.path.join("..", "MSMatch"))
from utils import net_builder

try:
    from onboard_prototype_utils import coarse_coregistration
except: # noqa E722
    raise ValueError("Impossible to find " "onboard_prototype_utils" ".")

import torch
import numpy as np
from torch.nn.functional import pad


class oboardDetectorProcessor:
    def __init__(
        self,
        satellite,
        detector_number,
        model_name="efficientnet-lite0",
        num_classes=2,
        in_channels=3,
        depth=28,
        widen_factor=2,
        leaky_slope=0.1,
        dropout=0.0,
        patch_overlap=0,
        device=torch.device("cpu"),
    ):
        self.satellite = satellite
        self.detector_number = detector_number

        if model_name is not None:
            _net_builder = net_builder(
                model_name,
                False,
                {
                    "depth": depth,
                    "widen_factor": widen_factor,
                    "leaky_slope": leaky_slope,
                    "dropRate": dropout,
                },
            )

            self.ai_model = _net_builder(
                num_classes=num_classes, in_channels=in_channels
            )
            self.ai_model.to(device)
        else:
            self.ai_model = None
        self.x_prev = None
        self.patch_overlap = patch_overlap
        self.x_coreg_pad = None
        self.h = 0
        self.v = 0
        self.finished = False
        self.device = device

    def flush(self):
        self.x_coreg_pad = None
        self.x_prev = None

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        load_model = checkpoint["eval_model"]
        self.ai_model.load_state_dict(load_model)

    def coarse_coregistration(self, x):
        x_coreg = coarse_coregistration(
            x,
            self.satellite,
            self.detector_number,
            X_prev=self.x_prev,
            crop_empty_pixels=True,
        )
        self.x_prev = x
        return x_coreg

    def init_patch_engine(self, x):
        n_v, n_h = int(x.shape[1] / 256), int(x.shape[2] / 256)
        self.x_coreg_pad = pad(
            x,
            (
                0,
                (256 - (x.shape[2] - n_h * 256)) % 256,
                0,
                (256 - (x.shape[1] - n_v * 256) % 256),
            ),
            "replicate",
        )
        self.h = 0
        self.v = 0
        self.finished = False

    def processing_finished(self):
        return self.finished

    def get_next_patch(self):
        x_patch = self.x_coreg_pad[:, self.v : self.v + 256, self.h : self.h + 256]

        if self.v < self.x_coreg_pad.shape[1] - (256 - self.patch_overlap):
            self.v += 256 - self.patch_overlap
        else:
            if self.h < self.x_coreg_pad.shape[2] - (256 - self.patch_overlap):
                self.h += 256 - self.patch_overlap
                self.v = 0
            else:
                self.finished = True
        return x_patch

    def get_output_shape(self):
        n_v_max = int(np.ceil(self.x_coreg_pad.shape[1] / (256 - self.patch_overlap)))
        n_h_max = int(np.ceil(self.x_coreg_pad.shape[2] / (256 - self.patch_overlap)))
        return [n_v_max, n_h_max]

    def process(self, x):
        # Applying coarse coregistration
        if self.ai_model is None:
            raise ValueError(
                "No model loaded. You need to specify a model before start the processing."
            )
        x_coreg = self.coarse_coregistration(x)
        self.init_patch_engine(x_coreg)
        # Size of the output mask
        n_v_max = int(np.ceil(self.x_coreg_pad.shape[1] / (256 - self.patch_overlap)))
        n_h_max = int(np.ceil(self.x_coreg_pad.shape[2] / (256 - self.patch_overlap)))
        y_demosaicked = torch.zeros([n_v_max, n_h_max]).to(x.device)
        n_v, n_h = 0, 0

        while not (self.processing_finished()):
            x_patch = self.get_next_patch()
            y_demosaicked[n_v, n_h] = self.ai_model(x_patch.unsqueeze(0)).max(1)[1]
            if n_v < n_v_max - 1:
                n_v += 1
            else:
                n_v = 0
                n_h += 1

        return y_demosaicked, x_coreg
