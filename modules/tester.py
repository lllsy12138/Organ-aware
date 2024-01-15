import logging
import os
from abc import abstractmethod

import cv2
import numpy as np
import spacy
import scispacy
import torch

from modules.utils import generate_heatmap


def check(gt, i):
    for keyword in list_keyword[i]:
        if gt.find(keyword) is not -1:
            return 1
    return 0


class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        self.model.eval()
        log = dict()
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - test' % epoch,
                      unit='it',
                      total=len(self.test_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag, reports_ids,
                                reports_masks) in enumerate(
                                    self.test_dataloader):
                    images = images.to(self.device)
                    for i in range(len(reports_ids)):
                        reports_ids[i] = reports_ids[i].to(self.device)
                        reports_masks[i] = reports_masks[i].to(self.device)
                    for i in range(self.num_cls):
                        tag[i] = tag[i].to(self.device)
                        tag[i] = torch.squeeze(tag[i], dim=1)

                    output, output2 = self.model(images, mode='sample')
                    for i in range(self.num_cls):
                        pre = torch.sigmoid(output[i].view(-1))
                        pre[pre >= 0.5] = 1
                        pre[pre < 0.5] = 0
                        acc_test[i] += float(torch.sum(pre == tag[i]))
                        sum_1_gt_test[i] += float(torch.sum(tag[i]))
                        sum_1_pre_test[i] += float(torch.sum(pre))
                        sum_1_right_test[i] += float(torch.sum(pre * tag[i]))

                    ground_truths = self.model.tokenizer.decode_batch(
                        reports_ids[-1][:, 1:].cpu().numpy())
                    reports = [''] * len(reports_ids[-1])
                    for j in range(len(output2)):
                        tmp = self.model.tokenizer.decode_batch(
                            output2[j].cpu().numpy())
                        for i in range(len(reports_ids[-1])):
                            if self.args.dataset_name == 'mimic_cxr' or check(
                                    ground_truths[i], j):
                                reports[i] = reports[i] + ' ' + tmp[i]

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    pbar.update()
                test_met = self.metric_ftns(
                    {i: [gt]
                     for i, gt in enumerate(test_gts)},
                    {i: [re]
                     for i, re in enumerate(test_res)})
                score = test_met
                f.write(
                    "\nbleu_1: %f ; bleu_2: %f ; bleu_3: %f ; bleu_4: %f ; METEOR: %f ; ROUGE: %f"
                    % (score['BLEU_1'], score['BLEU_2'], score['BLEU_3'],
                       score['BLEU_4'], score['METEOR'], score['ROUGE_L']))
                f.
                log.update(**{'val_' + k: v for k, v in test_met.items()})
                log.update(**{'test_' + k: v for k, v in test_met.items()})
        return log
