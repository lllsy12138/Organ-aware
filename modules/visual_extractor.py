import torch
import torch.nn as nn
import torchvision.models as models
from modules.resnet import resnet101

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        self.device = torch.cuda.current_device()
        self.model = nn.Sequential(*list(resnet101(pretrained=self.pretrained).children())[:-2]).to(self.device)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)  # B*2048*7*7
        # print(patch_feats.shape)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))  # B*2048*1*1
        # print(avg_feats.shape)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  # B*49*2048
        # print(patch_feats.shape)
        return patch_feats, avg_feats