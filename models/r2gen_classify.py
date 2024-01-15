import torch
import torch.nn as nn
import numpy as np
import copy
from modules.visual_extractor import VisualExtractor

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class R2GenModel(nn.Module):
    def __init__(self, args):
        super(R2GenModel, self).__init__()
        self.args = args  # 各种超参
        self.visual_extractor = VisualExtractor(args)  # 图像特征提取
        self.fc = clones(nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0, bias=True), 14).to('cuda:0')

    def forward(self, images):
        att_feats, fc_feats = self.visual_extractor(images)

        output=[]
        for i, l in enumerate(self.fc):
            x = l(fc_feats)
            x = x.squeeze(-1)
            x = x.squeeze(-1)
            output.append(x)
        return output

