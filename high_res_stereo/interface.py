import argparse
import cv2
from high_res_stereo.models import hsm
import numpy as np
import os
import pdb
import skimage.io
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from high_res_stereo.models.submodule import *
from high_res_stereo.models.submodule import disparityregression
from high_res_stereo.utils.eval import mkdir_p, save_pfm
from high_res_stereo.utils.preprocess import get_transform

from enum import Enum

class Level(Enum):
    COARSE = 3
    MEDIUM = 2
    FINE = 1


class HighResStereo(object):
    def __init__(self, weights_path: str, res_ratio=1.0, max_disparity=128, clean_up=True, level=Level.FINE):
        super(HighResStereo, self).__init__()
        self.weights_path: str = weights_path

        self.model = hsm(128, clean_up, level=level.value)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        pretrained_dict = torch.load(weights_path, map_location=torch.device("cpu"))

        pretrained_dict["state_dict"] = {
            k: v for k, v in pretrained_dict["state_dict"].items() if "disp" not in k
        }
        self.model.load_state_dict(pretrained_dict["state_dict"], strict=False)
        self.model.eval()
        self.res_ratio = res_ratio

        candidate_disparity = max_disparity * res_ratio // 64 * 64

        tmpdisp = int(candidate_disparity)

        if (candidate_disparity) > tmpdisp:
            self.model.module.maxdisp = tmpdisp + 64
        else:
            self.model.module.maxdisp = tmpdisp

        if self.model.module.maxdisp == 64:
            self.model.module.maxdisp = 128

        self.model.module.disp_reg8 = disparityregression(self.model.module.maxdisp, 16)
        self.model.module.disp_reg16 = disparityregression(self.model.module.maxdisp, 16)
        self.model.module.disp_reg32 = disparityregression(self.model.module.maxdisp, 32)
        self.model.module.disp_reg64 = disparityregression(self.model.module.maxdisp, 64)


        # dry run
        multip = 48
        imgL = np.zeros((24 * multip, 32 * multip, 3), "uint8")
        imgR = np.zeros((24 * multip, 32 * multip, 3), "uint8")

        self.predict(imgL, imgR)

        # self.model = nn.DataParallel(model, device_ids=[0])

    def predict(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        bgr=True,
    ):
        imgsize = left_image.shape[:2]
        if len(left_image.shape) == 2:
            cv2.cvtColor(left_image, cv2.COLOR_GRAY2RGB)
        elif bgr:
            cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

        if len(right_image.shape) == 2:
            cv2.cvtColor(right_image, cv2.COLOR_GRAY2RGB)
        elif bgr:
            cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        assert (
            left_image.shape == right_image.shape
        ), "Both images need to be the same shape"

        imgL, imgR, max_w, max_h = self.pre_process(left_image, right_image)


        with torch.no_grad():
            pred_disp, entropy = self.model(imgL, imgR)

        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad = max_h - imgsize[0]
        left_pad = max_w - imgsize[1]
        entropy = entropy[top_pad:, : pred_disp.shape[1] - left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:, : pred_disp.shape[1] - left_pad]

        pred_disp = cv2.resize(
            pred_disp / self.res_ratio,
            (imgsize[1], imgsize[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        invalid = np.logical_or(pred_disp == np.inf, pred_disp != pred_disp)
        pred_disp[invalid] = np.inf

        return pred_disp, entropy

    def pre_process(self, imgL_o, imgR_o):
        imgL_o = cv2.resize(
            imgL_o,
            None,
            fx=self.res_ratio,
            fy=self.res_ratio,
            interpolation=cv2.INTER_CUBIC,
        )
        imgR_o = cv2.resize(
            imgR_o,
            None,
            fx=self.res_ratio,
            fy=self.res_ratio,
            interpolation=cv2.INTER_CUBIC,
        )

        processed = get_transform()
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()

        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]:
            max_h += 64
        if max_w < imgL.shape[3]:
            max_w += 64

        top_pad = max_h - imgL.shape[2]
        left_pad = max_w - imgL.shape[3]
        imgL = np.lib.pad(
            imgL,
            ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)),
            mode="constant",
            constant_values=0,
        )
        imgR = np.lib.pad(
            imgR,
            ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)),
            mode="constant",
            constant_values=0,
        )

        imgL = Variable(torch.FloatTensor(imgL))
        imgR = Variable(torch.FloatTensor(imgR))

        return imgL, imgR, max_w, max_h


if __name__ == "__main__":
    model = HighResStereo("/home/jari/Downloads/final-768px.tar", res_ratio=1.5)
    imgL, imgR = cv2.imread("/home/jari/Work/improve_stereo/1/im0.png"), cv2.imread("/home/jari/Work/improve_stereo/1/im1.png")

    disp, entropy = model.predict(imgL, imgR)
