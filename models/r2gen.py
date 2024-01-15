import torch
import torch.nn as nn
import numpy as np
import copy
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder, Decoder_only, Encoder_edit, MultiHeadedAttention, \
    PositionwiseFeedForward, EncoderLayer, PositionalEncoding, clones
from einops import rearrange, repeat

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class R2GenModel_Base_iu_xray(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_Base_iu_xray, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])

        fc_feats_0 = fc_feats_0.squeeze().reshape(-1, att_feats_0.size(2))
        fc_feats_1 = fc_feats_1.squeeze().reshape(-1, att_feats_1.size(2))

        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

class R2GenModel_Base_mimic_cxr(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_Base_mimic_cxr, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

class R2GenModel_Multi_iu_xray(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_Multi_iu_xray, self).__init__()
        self.args = args  # 各种超参
        self.tokenizer = tokenizer  #
        self.visual_extractor = VisualExtractor(args)
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.num_heads, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        position = PositionalEncoding(args.d_model, args.dropout)
        self.encoder = Encoder_edit(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout), args.num_layers,
                                    args.d_model, args.drop_prob_lm, args.d_vf)
        self.decoders = clones(Decoder_only(args, tokenizer), 7)


    def __str__(self):  # 计算模型参数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        fc_feats = fc_feats.squeeze(-1).squeeze(-1)
        fc_feats, att_feats, memory, att_masks = self.encoder(fc_feats, att_feats)
        output_list = []
        if mode == 'train':
            for i, decode in enumerate(self.decoders):
                output2 = decode(memory, att_masks, targets[i], mode='forward')
                output_list.append(output2)
        elif mode == 'sample':
            for decode in self.decoders:
                output2, _ = decode(fc_feats, att_feats, memory, att_masks, mode='sample')
                output_list.append(output2)
        else:
            raise ValueError
        return output_list

class R2GenModel_Multi_mimic_cxr(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_Multi_mimic_cxr, self).__init__()
        self.args = args  # 各种超参
        self.tokenizer = tokenizer  #
        self.visual_extractor = VisualExtractor(args)
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.num_heads, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        position = PositionalEncoding(args.d_model, args.dropout)
        self.encoder = Encoder_edit(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout), args.num_layers,
                                    args.d_model, args.drop_prob_lm, args.d_vf)
        self.decoders = clones(Decoder_only(args, tokenizer), 8)

    def __str__(self):  # 计算模型参数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        fc_feats = fc_feats.squeeze(-1).squeeze(-1)
        fc_feats, att_feats, memory, att_masks = self.encoder(fc_feats, att_feats)
        output_list = []
        if mode == 'train':
            for i, decode in enumerate(self.decoders):
                output2 = decode(memory, att_masks, targets[i], mode='forward')
                output_list.append(output2)
        elif mode == 'sample':
            for decode in self.decoders:
                output2, _ = decode(fc_feats, att_feats, memory, att_masks, mode='sample')
                output_list.append(output2)
        else:
            raise ValueError
        return output_list

class R2GenModel_Base_Cls_iu_xray(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_Base_Cls_iu_xray, self).__init__()
        self.args = args  # 各种超参
        self.tokenizer = tokenizer  #
        self.visual_extractor = VisualExtractor(args)  # 图像特征提取
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.fc1 = clones(nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU(True)), 14)
        self.fc2 = clones(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True), 14)
        self.fc3 = nn.Linear(512 * self.args.num_cls, 2048)

    def __str__(self):  # 计算模型参数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        output = []
        used_0 = []
        used_1 = []

        for i, l in enumerate(self.fc1):
            x_0 = l(fc_feats_0)
            x_1 = l(fc_feats_1)
            used_0.append(x_0)
            used_1.append(x_1)

        for i, l in enumerate(self.fc2):
            x__0 = l(x_0)
            x__1 = l(x_1)
            x__0 = x__0.squeeze(-1).squeeze(-1)
            x__1 = x__1.squeeze(-1).squeeze(-1)
            output.append((x__0+x__1)/2)

        fc_feats = fc_feats.squeeze(-1).squeeze(-1)

        cls_0 = torch.cat((used_0[0], used_0[1], used_0[2], used_0[3], used_0[4], used_0[5], used_0[6], used_0[7],
                            used_0[8], used_0[9], used_0[10], used_0[11], used_0[12], used_0[13]), dim=1)
        cls_1 = torch.cat((used_1[0], used_1[1], used_1[2], used_1[3], used_1[4], used_1[5], used_1[6], used_1[7],
                            used_1[8], used_1[9], used_1[10], used_1[11], used_1[12], used_1[13]), dim=1)
        cls_0 = cls_0.squeeze(-1).squeeze(-1)
        cls_1 = cls_1.squeeze(-1).squeeze(-1)
        cls_feats_0 = self.fc3(cls_0)
        cls_feats_1 = self.fc3(cls_1)
        cls_feats_0 = cls_feats_0.unsqueeze(1)
        cls_feats_1 = cls_feats_1.unsqueeze(1)
        cls_feats = torch.cat((cls_feats_0, cls_feats_1), dim=1)

        att_feats = torch.cat((att_feats, cls_feats), dim=1)
        if mode == 'train':
            output2 = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output2, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, output2


class R2GenModel_Base_Cls_mimic_cxr(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_Base_Cls_mimic_cxr, self).__init__()
        self.args = args  # 各种超参
        self.tokenizer = tokenizer  #
        self.visual_extractor = VisualExtractor(args)  # 图像特征提取
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.fc1 = clones(nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU(True)), 14)
        self.fc2 = clones(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True), 14)
        self.fc3 = nn.Linear(512 * 14, 2048)

    def __str__(self):  # 计算模型参数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        output = []
        used = []

        for i, l in enumerate(self.fc1):
            x = l(fc_feats)
            used.append(x)

        for i, l in enumerate(self.fc2):
            x_ = l(x)
            x_ = x_.squeeze(-1).squeeze(-1)
            output.append(x_)

        fc_feats = fc_feats.squeeze(-1).squeeze(-1)

        cls = torch.cat((used[0], used[1], used[2], used[3], used[4], used[5], used[6], used[7],
                           used[8], used[9], used[10], used[11], used[12], used[13]), dim=1)
        cls = cls.squeeze(-1).squeeze(-1)
        cls_feats = self.fc3(cls)
        cls_feats = cls_feats.unsqueeze(1)

        att_feats = torch.cat((att_feats, cls_feats), dim=1)
        if mode == 'train':
            output2 = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output2, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output, output2

class R2GenModel_Multi_Cls_iu_xray(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_Multi_Cls_iu_xray, self).__init__()
        self.args = args  # 各种超参
        self.tokenizer = tokenizer  #
        self.visual_extractor = VisualExtractor(args)
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.num_heads, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        position = PositionalEncoding(args.d_model, args.dropout)
        self.encoder = Encoder_edit(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout), args.num_layers,
                                    args.d_model, args.drop_prob_lm, args.d_vf)
        self.decoders = clones(Decoder_only(args, tokenizer), 7)
        
        self.cls_tokens_0 = nn.Parameter(torch.randn(1, 1, 2048))
        self.cls_tokens_1 = nn.Parameter(torch.randn(1, 1, 2048))
        self.dropout = nn.Dropout(0.1)

        self.to_latent = nn.Identity()

        self.mlp_heads = clones(nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.Linear(args.d_model, 1)
        ), self.args.num_cls)

    def __str__(self):  # 计算模型参数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        #att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        b, n, _ = att_feats_0.shape

        cls_tokens_0 = repeat(self.cls_tokens_0, '() n d -> b n d', b = b)
        cls_tokens_1 = repeat(self.cls_tokens_1, '() n d -> b n d', b = b)
        att_feats = torch.cat((cls_tokens_0, att_feats_0, cls_tokens_1, att_feats_1), dim=1)

        att_feats = self.dropout(att_feats)

        fc_feats, att_feats, memory, att_masks = self.encoder(fc_feats, att_feats)

        x = memory.mean(dim=1)
        output = []

        for i, l in enumerate(self.mlp_heads):
            output.append(l(x))

        output_list = []
        if mode == 'train':
            for i, decode in enumerate(self.decoders):
                output2 = decode(memory, att_masks, targets[i], mode='forward')
                output_list.append(output2)
        elif mode == 'sample':
            for decode in self.decoders:
                output2, _ = decode(fc_feats, att_feats, memory, att_masks, mode='sample')
                output_list.append(output2)
        else:
            raise ValueError
        return output, output_list

class R2GenModel_Multi_Cls_mimic_cxr(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_Multi_Cls_mimic_cxr, self).__init__()
        self.args = args  # 各种超参
        self.tokenizer = tokenizer  #
        self.visual_extractor = VisualExtractor(args)
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.num_heads, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        position = PositionalEncoding(args.d_model, args.dropout)
        self.encoder = Encoder_edit(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout), args.num_layers,
                                    args.d_model, args.drop_prob_lm, args.d_vf)
        self.decoders = clones(Decoder_only(args, tokenizer), 8)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 2048))
        self.dropout = nn.Dropout(0.1)

        self.to_latent = nn.Identity()

        self.mlp_heads = clones(nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.Linear(args.d_model, 1)
        ), self.args.num_cls)

    def __str__(self):  # 计算模型参数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        
        b, n, _ = att_feats.shape   # B*49*2048

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        att_feats = torch.cat((cls_tokens, att_feats), dim=1)   # B*50*2048

        att_feats = self.dropout(att_feats)

        fc_feats, att_feats, memory, att_masks = self.encoder(fc_feats, att_feats)  
        #att_feats是embedding之后的
        #memory即encode的结果
        x = memory.mean(dim=1)
        output = []
        #x是classification token
        for i, l in enumerate(self.mlp_heads):
            output.append(l(x))

        output_list = []
        if mode == 'train':
            for i, decode in enumerate(self.decoders):
                output2 = decode(memory, att_masks, targets[i], mode='forward')
                output_list.append(output2)
        elif mode == 'sample':
            for decode in self.decoders:
                output2, _ = decode(fc_feats, att_feats, memory, att_masks, mode='sample')
                output_list.append(output2)
        else:
            raise ValueError
        return output, output_list

class R2GenModel_Multi_Cls_two_iu_xray(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_Multi_Cls_two_iu_xray, self).__init__()
        self.args = args  # 各种超参
        self.tokenizer = tokenizer  #
        self.visual_extractor = VisualExtractor(args)
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.num_heads, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        position = PositionalEncoding(args.d_model, args.dropout)
        self.encoder = Encoder_edit(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout), args.num_layers,
                                    args.d_model, args.drop_prob_lm, args.d_vf)
        self.decoders = clones(Decoder_only(args, tokenizer), 8)
        
        self.cls_tokens_0 = nn.Parameter(torch.randn(1, 1, 2048))
        self.cls_tokens_1 = nn.Parameter(torch.randn(1, 1, 2048))
        self.dropout = nn.Dropout(0.1)

        self.to_latent = nn.Identity()

        self.mlp_heads = clones(nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.Linear(args.d_model, 1)
        ), self.args.num_cls)

    def __str__(self):  # 计算模型参数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        #att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        b, n, _ = att_feats_0.shape

        cls_tokens_0 = repeat(self.cls_tokens_0, '() n d -> b n d', b = b)
        cls_tokens_1 = repeat(self.cls_tokens_1, '() n d -> b n d', b = b)
        att_feats = torch.cat((cls_tokens_0, att_feats_0, cls_tokens_1, att_feats_1), dim=1)

        att_feats = self.dropout(att_feats)

        fc_feats, att_feats, memory, att_masks = self.encoder(fc_feats, att_feats)

        x = memory.mean(dim=1)
        output = []

        for i, l in enumerate(self.mlp_heads):
            output.append(l(x))

        output_list = []
        if mode == 'train':
            for i, decode in enumerate(self.decoders):
                output2 = decode(memory, att_masks, targets[i], mode='forward')
                output_list.append(output2)
        elif mode == 'sample':
            for decode in self.decoders:
                output2, _ = decode(fc_feats, att_feats, memory, att_masks, mode='sample')
                output_list.append(output2)
        else:
            raise ValueError
        return output, output_list

class R2GenModel_Multi_Cls_two_mimic_cxr(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_Multi_Cls_two_mimic_cxr, self).__init__()
        self.args = args  # 各种超参
        self.tokenizer = tokenizer  #
        self.visual_extractor = VisualExtractor(args)
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.num_heads, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        position = PositionalEncoding(args.d_model, args.dropout)
        self.encoder = Encoder_edit(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout), args.num_layers,
                                    args.d_model, args.drop_prob_lm, args.d_vf)
        self.decoders = clones(Decoder_only(args, tokenizer), 9)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 2048))
        self.dropout = nn.Dropout(0.1)

        self.to_latent = nn.Identity()

        self.mlp_heads = clones(nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.Linear(args.d_model, 1)
        ), self.args.num_cls)

    def __str__(self):  # 计算模型参数量
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        
        b, n, _ = att_feats.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        att_feats = torch.cat((cls_tokens, att_feats), dim=1)

        att_feats = self.dropout(att_feats)

        fc_feats, att_feats, memory, att_masks = self.encoder(fc_feats, att_feats)

        x = memory.mean(dim=1)
        output = []

        for i, l in enumerate(self.mlp_heads):
            output.append(l(x))

        output_list = []
        if mode == 'train':
            for i, decode in enumerate(self.decoders):
                output2 = decode(memory, att_masks, targets[i], mode='forward')
                output_list.append(output2)
        elif mode == 'sample':
            for decode in self.decoders:
                output2, _ = decode(fc_feats, att_feats, memory, att_masks, mode='sample')
                output_list.append(output2)
        else:
            raise ValueError
        return output, output_list


class R2GenModel_edit(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_edit, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.num_heads, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        position = PositionalEncoding(args.d_model, args.dropout)
        self.visual_extractor = VisualExtractor(args)
        self.encoder = Encoder_edit(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout), args.num_layers, args.d_model, args.drop_prob_lm, args.d_vf)
        self.decoders = clones(Decoder_only(args, tokenizer), 8)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        fc_feats, att_feats, memory, att_masks = self.encoder(fc_feats, att_feats)
        output_list = []
        if mode == 'train':
            for i, decode in enumerate(self.decoders):
                output = decode(memory, att_masks, targets[i], mode='forward')
                output_list.append(output)
        elif mode == 'sample':
            for decode in self.decoders:
                output, _ = decode(fc_feats, att_feats, memory, att_masks, mode='sample')
                output_list.append(output)
        else:
            raise ValueError
        return output_list

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)

        fc_feats, att_feats, memory, att_masks = self.encoder(fc_feats, att_feats)

        output_list = []
        if mode == 'train':
            for i, decode in enumerate(self.decoders):
                output = decode(memory, att_masks, targets[i], mode='forward')
                output_list.append(output)
        elif mode == 'sample':
            for decode in self.decoders:
                output, _ = decode(fc_feats, att_feats, memory, att_masks, mode='sample')
                output_list.append(output)
        else:
            raise ValueError
        return output_list

class R2GenModel_plus(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel_plus, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.num_heads, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        position = PositionalEncoding(args.d_model, args.dropout)
        self.visual_extractor = VisualExtractor(args)
        self.encoder = Encoder_edit(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout), args.num_layers, args.d_model, args.drop_prob_lm, args.d_vf)
        self.decoders = clones(Decoder_only_edit(args, tokenizer), 8)
        self.final_decoder = Decoder_only(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        fc_feats, att_feats, memory, att_masks = self.encoder(fc_feats, att_feats)
        output_list = []
        out_list = []
        if mode == 'train':
            for i, decode in enumerate(self.decoders):
                output,out = decode(memory, att_masks, targets[i], mode='forward')
                output_list.append(output)
                out_list.append(out)
        elif mode == 'sample':
            for decode in self.decoders:
                output, _ = decode(fc_feats, att_feats, memory, att_masks, mode='sample')
                output_list.append(output)
        else:
            raise ValueError
        return output_list

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)

        fc_feats, att_feats, memory, att_masks = self.encoder(fc_feats, att_feats)

        output_list = []
        if mode == 'train':
            for i, decode in enumerate(self.decoders):
                output = decode(memory, att_masks, targets[i], mode='forward')
                output_list.append(output)
        elif mode == 'sample':
            for decode in self.decoders:
                output, _ = decode(fc_feats, att_feats, memory, att_masks, mode='sample')
                output_list.append(output)
        else:
            raise ValueError
        return output_list
