import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


class LanguageModelCriterion_edit(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion_edit, self).__init__()

    def forward(self, input, target, mask,mask_sum):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output)
        #output_sum = torch.sum(mask)

        return output / (mask_sum + 0.001)

def compute_loss_edit(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion_edit()
    mask_sum = torch.sum(reports_masks[0][:, 1:])
    for i in range(1, len(output)):
        mask_sum = mask_sum + torch.sum(reports_masks[i][:, 1:])

    loss = criterion(output[0], reports_ids[0][:, 1:], reports_masks[0][:, 1:],mask_sum).mean()
    for i in range(1, len(output)):
        loss = loss + criterion(output[i], reports_ids[i][:, 1:], reports_masks[i][:, 1:],mask_sum).mean()
    return loss.mean()
    
def bce2d(pred, gt, reduction='mean'):
    pred = pred.view(-1)
    gt = gt.view(-1).float()
    pos = torch.eq(gt, 1).float()
    neg = torch.eq(gt, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha_pos = num_neg / num_total
    alpha_neg = num_pos / num_total
    weights = alpha_pos * pos + 1.1 * alpha_neg * neg
    return F.binary_cross_entropy_with_logits(pred, gt, weights, reduction = reduction)
