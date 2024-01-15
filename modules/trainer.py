import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from modules.loss import bce2d


def is_master_proc(num_gpus=8):
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True
class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        _ , device_ids = self._prepare_device(args.n_gpu)
        self.device = torch.cuda.current_device()
        self.model = model.to(self.device)
        #if len(device_ids) > 1:
        #    self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def is_master_proc(num_gpus=8):
        if torch.distributed.is_initialized():
            return dist.get_rank() % num_gpus == 0
        else:
            return True
    
    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))

class Trainer_Base(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer_Base, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        f = open('%s.txt' % self.args.save_dir, 'a+')
        f.write("\n+++++++++++++++++++++++++++++++++ epoch %d +++++++++++++++++++++++++++++++++" % epoch)
        f.write("\ninformation of training: ")
        train_loss = 0
        self.model.train()
        print(len(self.train_dataloader))
        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device)
                output = self.model(images, reports_ids, mode='train')
                loss = self.criterion(output, reports_ids, reports_masks)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update()
            log = {'train_loss': train_loss / len(self.train_dataloader)}

        f.write("\ninformation of val: ")
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            with tqdm(desc='Epoch %d - val' % epoch, unit='it', total=len(self.val_dataloader)) as pbar:
                for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                    images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                        self.device), reports_masks.to(self.device)
                    output = self.model(images, mode='sample')
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    pbar.update()
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                        {i: [re] for i, re in enumerate(val_res)})
                log.update(**{'val_' + k: v for k, v in val_met.items()})

        f.write("\ninformation of test: ")
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - test' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
                for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                    images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                        self.device), reports_masks.to(self.device)
                    output = self.model(images, mode='sample')
                    reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    pbar.update()
                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})

                log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log

list_keyword = [["heart","cardiomediastinal","cardiac","cardiomegaly","cardial","cardio","mediastinum","mediastinal","hemidiaphragm","diaphragm","hernia"],
        ["lung","pulmonary","pneumonia","granulomatous","granuloma","emphysema","hyperinflated ",
    "atelectasis","edema","nodules","masses", "lesion", "lobe","opacity","opacit","opacification"],
        ["bone","bony","osseous","skeletal","spine","spondylosis","osseus","fracture","rib","vertebrae"],
        ["pleural","pneumothorax","effusion","thickening"],
        ["airspace","air space","air"],
        ["thoracic aorta","aorta","aortic","vascular"],
        ["trachea","tube","picc","catheter","course","pacemaker","port-a-cath",
    "device","clip","pacer","stent"]]

def check(gt, i):
    for keyword in list_keyword[i]:
        if gt.find(keyword) != -1:
            return 1
    return 0

def output_func(list_val):
    #return "\n1: %f ; 2: %f ; 3: %f ; 4: %f ; 5: %f ; 6: %f; 7: %f\n 8: %f ; 9: %f ; 10: %f ; 11: %f ; 12: %f ; 13: %f; 14: %f" % (
    #list_val[0], list_val[1], list_val[2], list_val[3], list_val[4], list_val[5], list_val[6], list_val[7], list_val[8],
    #list_val[9], list_val[10], list_val[11], list_val[12], list_val[13])
    return "\n1: %f ; 2: %f ; 3: %f ; 4: %f ; 5: %f ; 6: %f; 7: %f\n" % (
    list_val[0], list_val[1], list_val[2], list_val[3], list_val[4], list_val[5], list_val[6])

def F1_score(list_gt, list_pre, list_right, num_cls):
    TP = [0.0 for i in range(num_cls)]
    FP = [0.0 for i in range(num_cls)]
    TN = [0.0 for i in range(num_cls)]
    FN = [0.0 for i in range(num_cls)]
    precision = [0.0 for i in range(num_cls)]
    recall = [0.0 for i in range(num_cls)]
    list_F1 = [0.0 for i in range(num_cls)]
    TP = list_right
    for i in range(num_cls):
        FP[i] = list_pre[i] - list_right[i]
        FN[i] = list_gt[i] - list_right[i]
        precision[i] = (TP[i]) / (TP[i] + FP[i] + 0.001)
        recall[i] = (TP[i]) / (TP[i] + FN[i] + 0.001)
        list_F1[i] = (2.0 * TP[i]) / (2.0 * TP[i] + FP[i] + FN[i] + 0.001)
    return list_F1

class BaseTrainer_classify(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        print(self.device, device_ids)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)
            self._save_checkpoint(epoch, save_best=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint_%d.pth'%epoch)
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

class Trainer_Multi_Cls(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer_Multi_Cls, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_cls = 7

    def _train_epoch(self, epoch):
        f = open('%s.txt' % self.args.save_dir, 'a+')
        f.write("\n+++++++++++++++++++++++++++++++++ epoch %d +++++++++++++++++++++++++++++++++" % epoch)
        train_loss = 0
        self.model.train()
        print(self.device)
        print(next(self.model.parameters()).device)
        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                for i in range(len(reports_ids)):
                    reports_ids[i] = reports_ids[i].to(self.device)
                    reports_masks[i] = reports_masks[i].to(self.device)

                for i in range(self.num_cls):
                    tag[i] = tag[i].to(self.device)
                    tag[i] = torch.squeeze(tag[i], dim=1)
                output, output2 = self.model(images, reports_ids, mode='train')
                loss1 = 0
                for i in range(self.num_cls):
                    loss1 += bce2d(output[i], tag[i])
                loss2 = self.criterion(output2, reports_ids, reports_masks)
                loss = loss1 * 0.5 + loss2
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update()
            log = {'train_loss': train_loss / len(self.train_dataloader)}

        num_val = 2130 if self.args.dataset_name == 'mimic_cxr' else 296
        num_test = 3858 if self.args.dataset_name == 'mimic_cxr' else 590
        acc_val = [0.0 for i in range(self.num_cls)]

        sum_1_gt_val = [0.0 for i in range(self.num_cls)]
        sum_1_pre_val = [0.0 for i in range(self.num_cls)]
        sum_1_right_val = [0.0 for i in range(self.num_cls)]

        acc_test = [0.0 for i in range(self.num_cls)]

        sum_1_gt_test = [0.0 for i in range(self.num_cls)]
        sum_1_pre_test = [0.0 for i in range(self.num_cls)]
        sum_1_right_test = [0.0 for i in range(self.num_cls)]
        """
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            with tqdm(desc='Epoch %d - val' % epoch, unit='it', total=len(self.val_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                    images = images.to(self.device)
                    for i in range(len(reports_ids)):
                        reports_ids[i] = reports_ids[i].to(self.device)
                        reports_masks[i] = reports_masks[i].to(self.device)
                    for i in range(14):
                        tag[i] = tag[i].to(self.device)
                        tag[i] = torch.squeeze(tag[i], dim=1)

                    output, output2 = self.model(images, mode='sample')
                    for i in range(14):
                        pre = torch.sigmoid(output[i].view(-1))
                        pre[pre>=0.5] = 1
                        pre[pre<0.5] = 0
                        acc_val[i] += float(torch.sum(pre == tag[i]))
                        sum_1_gt_val[i] += float(torch.sum(tag[i]))
                        sum_1_pre_val[i] += float(torch.sum(pre))
                        sum_1_right_val[i] += float(torch.sum(pre * tag[i]))

                    reports = [''] * len(reports_ids[-1])
                    for j in range(len(output2)):
                        tmp = self.model.tokenizer.decode_batch(output2[j].cpu().numpy())
                        for i in range(len(reports_ids[-1])):
                            reports[i] = reports[i] + tmp[i][:len(tmp[i])]

                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[-1][:, 1:].cpu().numpy())
                    for i in range(len(reports_ids[-1])):
                        reports[i] = reports[i][:len(reports[i])]
                        ground_truths[i] = ground_truths[i][:len(ground_truths[i])]
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                    pbar.update()
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                           {i: [re] for i, re in enumerate(val_res)})
                score = val_met
                f.write("\nbleu_1: %f ; bleu_2: %f ; bleu_3: %f ; bleu_4: %f ; METEOR: %f ; ROUGE: %f" % (
                    score['BLEU_1'], score['BLEU_2'], score['BLEU_3'], score['BLEU_4'], score['METEOR'],
                    score['ROUGE_L']))
                log.update(**{'val_' + k: v for k, v in val_met.items()})
        for i in range(14):
            acc_val[i] /= num_val
        f.write("\ninformation of val: ")
        f.write("\nacc: ")
        f.write(output_func(acc_val))
        f.write("\nF1: ")
        F1_val = F1_score(sum_1_gt_val, sum_1_pre_val, sum_1_right_val)
        f.write(output_func(F1_val))
        """
        self.model.eval()
        reports1=[]
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - test' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.test_dataloader):
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
                        pre[pre>=0.5] = 1
                        pre[pre<0.5] = 0
                        acc_test[i] += float(torch.sum(pre == tag[i]))
                        sum_1_gt_test[i] += float(torch.sum(tag[i]))
                        sum_1_pre_test[i] += float(torch.sum(pre))
                        sum_1_right_test[i] += float(torch.sum(pre * tag[i]))

                    output, output2  = self.model(images, mode='sample')
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[-1][:, 1:].cpu().numpy())
                    reports = [''] * len(reports_ids[-1])
                    for j in range(len(output2)):
                        tmp = self.model.tokenizer.decode_batch(output2[j].cpu().numpy())
                        for i in range(len(reports_ids[-1])):
                            if self.args.dataset_name == 'mimic_cxr' or check(ground_truths[i],j):
                                reports[i] = reports[i] + ' ' + tmp[i]

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    pbar.update()
                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                score = test_met
                f.write("\nbleu_1: %f ; bleu_2: %f ; bleu_3: %f ; bleu_4: %f ; METEOR: %f ; ROUGE: %f" % (
                    score['BLEU_1'], score['BLEU_2'], score['BLEU_3'], score['BLEU_4'], score['METEOR'],
                    score['ROUGE_L']))
                log.update(**{'val_' + k: v for k, v in test_met.items()})
                log.update(**{'test_' + k: v for k, v in test_met.items()})

        for i in range(self.num_cls):
            acc_test[i] /= num_test
        f.write("\ninformation of test: ")
        f.write("\nacc: ")
        f.write(output_func(acc_test))
        f.write("\nF1: ")
        F1_test = F1_score(sum_1_gt_test, sum_1_pre_test, sum_1_right_test,self.num_cls)
        f.write(output_func(F1_test))

        self.lr_scheduler.step()
        return log

class Trainer_Multi(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer_Multi, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_cls = 7

    def _train_epoch(self, epoch):
        f = open('%s.txt' % self.args.save_dir, 'a+')
        f.write("\n+++++++++++++++++++++++++++++++++ epoch %d +++++++++++++++++++++++++++++++++" % epoch)
        train_loss = 0
        self.model.train()

        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.train_dataloader):

                images = images.to(self.device)
                for i in range(len(reports_ids)):
                    reports_ids[i] = reports_ids[i].to(self.device)
                    reports_masks[i] = reports_masks[i].to(self.device)

                output2 = self.model(images, reports_ids, mode='train')
                loss2 = self.criterion(output2, reports_ids, reports_masks)
                loss = loss2
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update()
            log = {'train_loss': train_loss / len(self.train_dataloader)}

        """
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            with tqdm(desc='Epoch %d - val' % epoch, unit='it', total=len(self.val_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                    images = images.to(self.device)
                    for i in range(len(reports_ids)):
                        reports_ids[i] = reports_ids[i].to(self.device)
                        reports_masks[i] = reports_masks[i].to(self.device)

                    output2 = self.model(images, mode='sample')

                    reports = [''] * len(reports_ids[-1])
                    for j in range(len(output2)):
                        tmp = self.model.tokenizer.decode_batch(output2[j].cpu().numpy())
                        for i in range(len(reports_ids[-1])):
                            reports[i] = reports[i] + tmp[i][:len(tmp[i])]

                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[-1][:, 1:].cpu().numpy())
                    for i in range(len(reports_ids[-1])):
                        reports[i] = reports[i][:len(reports[i])]
                        ground_truths[i] = ground_truths[i][:len(ground_truths[i])]
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                    pbar.update()
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                           {i: [re] for i, re in enumerate(val_res)})
                score = val_met
                f.write("\nbleu_1: %f ; bleu_2: %f ; bleu_3: %f ; bleu_4: %f ; METEOR: %f ; ROUGE: %f" % (
                    score['BLEU_1'], score['BLEU_2'], score['BLEU_3'], score['BLEU_4'], score['METEOR'],
                    score['ROUGE_L']))
                log.update(**{'val_' + k: v for k, v in val_met.items()})
        for i in range(14):
            acc_val[i] /= num_val
        f.write("\ninformation of val: ")
        f.write("\nacc: ")
        f.write(output_func(acc_val))
        f.write("\nF1: ")
        F1_val = F1_score(sum_1_gt_val, sum_1_pre_val, sum_1_right_val)
        f.write(output_func(F1_val))
        """
        f.write("\ninformation of test: ")
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - test' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                    images = images.to(self.device)
                    for i in range(len(reports_ids)):
                        reports_ids[i] = reports_ids[i].to(self.device)
                        reports_masks[i] = reports_masks[i].to(self.device)

                    output2 = self.model(images, mode='sample')

                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[-1][:, 1:].cpu().numpy())
                    reports = [''] * len(reports_ids[-1])
                    for j in range(len(output2)):
                        tmp = self.model.tokenizer.decode_batch(output2[j].cpu().numpy())
                        for i in range(len(reports_ids[-1])):
                            if self.args.dataset_name == 'iu_xray' and check(ground_truths[i],j):
                                reports[i] = reports[i] + ' ' + tmp[i]
                    
                    test_res.extend(reports)
                    test_gts.extend(ground_truths)

                    pbar.update()
                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                score = test_met
                f.write("\nbleu_1: %f ; bleu_2: %f ; bleu_3: %f ; bleu_4: %f ; METEOR: %f ; ROUGE: %f" % (
                    score['BLEU_1'], score['BLEU_2'], score['BLEU_3'], score['BLEU_4'], score['METEOR'],
                    score['ROUGE_L']))
                log.update(**{'val_' + k: v for k, v in test_met.items()})
                log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()
        return log

class Trainer_Base_Cls(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer_Base_Cls, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_cls = 8

    def _train_epoch(self, epoch):
        f = open('%s.txt' % self.args.save_dir, 'a+')
        f.write("\n+++++++++++++++++++++++++++++++++ epoch %d +++++++++++++++++++++++++++++++++" % epoch)
        train_loss = 0
        self.model.train()

        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                for i in range(len(reports_ids)):
                    reports_ids[i] = reports_ids[i].to(self.device)
                    reports_masks[i] = reports_masks[i].to(self.device)

                for i in range(self.num_cls):
                    tag[i] = tag[i].to(self.device)
                    tag[i] = torch.squeeze(tag[i], dim=1)
                output, output2 = self.model(images, reports_ids[-1], mode='train')
                loss1 = 0
                for i in range(self.num_cls):
                    loss1 += bce2d(output[i], tag[i])
                loss2 = self.criterion(output2, reports_ids[-1], reports_masks[-1])
                loss = loss1 * 0.5 + loss2
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update()
            log = {'train_loss': train_loss / len(self.train_dataloader)}

        num_val = 2130
        num_test = 3858
        acc_val = [0.0 for i in range(self.num_cls)]

        sum_1_gt_val = [0.0 for i in range(self.num_cls)]
        sum_1_pre_val = [0.0 for i in range(self.num_cls)]
        sum_1_right_val = [0.0 for i in range(self.num_cls)]

        acc_test = [0.0 for i in range(self.num_cls)]

        sum_1_gt_test = [0.0 for i in range(self.num_cls)]
        sum_1_pre_test = [0.0 for i in range(self.num_cls)]
        sum_1_right_test = [0.0 for i in range(self.num_cls)]
        """
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            with tqdm(desc='Epoch %d - val' % epoch, unit='it', total=len(self.val_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                    images = images.to(self.device)
                    for i in range(len(reports_ids)):
                        reports_ids[i] = reports_ids[i].to(self.device)
                        reports_masks[i] = reports_masks[i].to(self.device)
                    for i in range(14):
                        tag[i] = tag[i].to(self.device)
                        tag[i] = torch.squeeze(tag[i], dim=1)

                    output, output2 = self.model(images, mode='sample')
                    for i in range(14):
                        pre = torch.sigmoid(output[i].view(-1))
                        pre[pre>=0.5] = 1
                        pre[pre<0.5] = 0
                        acc_val[i] += float(torch.sum(pre == tag[i]))
                        sum_1_gt_val[i] += float(torch.sum(tag[i]))
                        sum_1_pre_val[i] += float(torch.sum(pre))
                        sum_1_right_val[i] += float(torch.sum(pre * tag[i]))

                    reports = self.model.tokenizer.decode_batch(output2.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[-1][:, 1:].cpu().numpy())
                    for i in range(len(reports_ids[-1])):
                        reports[i] = reports[i][:len(reports[i])]
                        ground_truths[i] = ground_truths[i][:len(ground_truths[i])]
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)
                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                           {i: [re] for i, re in enumerate(val_res)})
                score = val_met
                f.write("\nbleu_1: %f ; bleu_2: %f ; bleu_3: %f ; bleu_4: %f ; METEOR: %f ; ROUGE: %f" % (
                    score['BLEU_1'], score['BLEU_2'], score['BLEU_3'], score['BLEU_4'], score['METEOR'],
                    score['ROUGE_L']))
                log.update(**{'val_' + k: v for k, v in val_met.items()})
        for i in range(14):
            acc_val[i] /= num_val
        f.write("\ninformation of val: ")
        f.write("\nacc: ")
        f.write(output_func(acc_val))
        f.write("\nF1: ")
        F1_val = F1_score(sum_1_gt_val, sum_1_pre_val, sum_1_right_val)
        f.write(output_func(F1_val))
        """
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - test' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.test_dataloader):
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
                        pre[pre>=0.5] = 1
                        pre[pre<0.5] = 0
                        acc_test[i] += float(torch.sum(pre == tag[i]))
                        sum_1_gt_test[i] += float(torch.sum(tag[i]))
                        sum_1_pre_test[i] += float(torch.sum(pre))
                        sum_1_right_test[i] += float(torch.sum(pre * tag[i]))

                    reports = self.model.tokenizer.decode_batch(output2.cpu().numpy())
                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[-1][:, 1:].cpu().numpy())
                    for i in range(len(reports_ids[-1])):
                        reports[i] = reports[i][:len(reports[i])]
                        ground_truths[i] = ground_truths[i][:len(ground_truths[i])]
                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    pbar.update()
                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                score = test_met
                f.write("\nbleu_1: %f ; bleu_2: %f ; bleu_3: %f ; bleu_4: %f ; METEOR: %f ; ROUGE: %f" % (
                    score['BLEU_1'], score['BLEU_2'], score['BLEU_3'], score['BLEU_4'], score['METEOR'],
                    score['ROUGE_L']))

                log.update(**{'val_' + k: v for k, v in test_met.items()})
                log.update(**{'test_' + k: v for k, v in test_met.items()})

        for i in range(self.num_cls):
            acc_test[i] /= num_test
        f.write("\ninformation of test: ")
        f.write("\nacc: ")
        f.write(output_func(acc_test))
        f.write("\nF1: ")
        F1_test = F1_score(sum_1_gt_test, sum_1_pre_test, sum_1_right_test, self.num_cls)
        f.write(output_func(F1_test))

        self.lr_scheduler.step()
        return log

class Trainer_classify(BaseTrainer_classify):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer_classify, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_cls

    def _train_epoch(self, epoch):
        f = open('%s.txt' % self.args.save_dir, 'a+')
        f.write("\n+++++++++++++++++++++++++++++++++ epoch %d +++++++++++++++++++++++++++++++++" % epoch)
        train_loss = 0
        self.model.train()

        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for batch_idx, (images_id, images, tag) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                for i in range(14):
                    tag[i] = tag[i].to(self.device)
                    tag[i] = torch.squeeze(tag[i], dim=1)
                output = self.model(images)
                loss = 0
                for i in range(self.num_cls):
                    loss += bce2d(output[i], tag[i])
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update()
            log = {'train_loss': train_loss / len(self.train_dataloader)}

        num_val = 234# 2130
        num_test = 234# 3858
        acc_val = [0.0 for i in range(self.num_cls)]

        sum_1_gt_val = [0.0 for i in range(self.num_cls)]
        sum_1_pre_val = [0.0 for i in range(self.num_cls)]
        sum_1_right_val = [0.0 for i in range(self.num_cls)]

        acc_test = [0.0 for i in range(self.num_cls)]

        sum_1_gt_test = [0.0 for i in range(self.num_cls)]
        sum_1_pre_test = [0.0 for i in range(self.num_cls)]
        sum_1_right_test = [0.0 for i in range(self.num_cls)]

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            with tqdm(desc='Epoch %d - val' % epoch, unit='it', total=len(self.val_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag) in enumerate(self.val_dataloader):
                    images = images.to(self.device)
                    for i in range(self.num_cls):
                        tag[i] = tag[i].to(self.device)
                        tag[i] = torch.squeeze(tag[i], dim=1)

                    output = self.model(images)
                    for i in range(self.num_cls):
                        pre = torch.sigmoid(output[i].view(-1))#.detach().cpu().numpy()
                        pre[pre>=0.5] = 1
                        pre[pre<0.5] = 0
                        acc_val[i] += float(torch.sum(pre == tag[i]))
                        sum_1_gt_val[i] += float(torch.sum(tag[i]))
                        sum_1_pre_val[i] += float(torch.sum(pre))
                        sum_1_right_val[i] += float(torch.sum(pre * tag[i]))
                    pbar.update()
        for i in range(14):
            acc_val[i] /= num_val
        f.write("\ninformation of val: ")
        f.write("\nacc: ")
        f.write(output_func(acc_val))
        f.write("\nF1: ")
        F1_val = F1_score(sum_1_gt_val, sum_1_pre_val, sum_1_right_val)
        f.write(output_func(F1_val))

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - test' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag) in enumerate(self.test_dataloader):
                    images = images.to(self.device)
                    for i in range(self.num_cls):
                        tag[i] = tag[i].to(self.device)
                        tag[i] = torch.squeeze(tag[i], dim=1)

                    output = self.model(images)
                    for i in range(self.num_cls):
                        pre = torch.sigmoid(output[i].view(-1))
                        pre[pre>=0.5] = 1
                        pre[pre<0.5] = 0
                        acc_test[i] += float(torch.sum(pre == tag[i]))
                        sum_1_gt_test[i] += float(torch.sum(tag[i]))
                        sum_1_pre_test[i] += float(torch.sum(pre))
                        sum_1_right_test[i] += float(torch.sum(pre * tag[i]))

                    pbar.update()
        for i in range(self.num_cls):
            acc_test[i] /= num_test
        f.write("\ninformation of test: ")
        f.write("\nacc: ")
        f.write(output_func(acc_test))
        f.write("\nF1: ")
        F1_test = F1_score(sum_1_gt_test, sum_1_pre_test, sum_1_right_test)
        f.write(output_func(F1_test))

        self.lr_scheduler.step()

class Trainer_Multi_Cls_ABL(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer_Multi_Cls_ABL, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_cls = 7

        
    def _train_epoch(self, epoch):
        f = open('%s.txt' % self.args.save_dir, 'a+')
        f2 = open('results/mimic_cxr_multi_cls_ABLb.txt' , 'a+')
        f.write("\n+++++++++++++++++++++++++++++++++ epoch %d +++++++++++++++++++++++++++++++++" % epoch)
        train_loss = 0
        self.model.train()
        print(self.device)
        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(self.train_dataloader),disable=not is_master_proc()) as pbar:
            for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                for i in range(len(reports_ids)):
                    reports_ids[i] = reports_ids[i].to(self.device)
                    reports_masks[i] = reports_masks[i].to(self.device)

                for i in range(self.num_cls):
                    tag[i] = tag[i].to(self.device)
                    tag[i] = torch.squeeze(tag[i], dim=1)
                output, output2 = self.model(images, reports_ids, mode='train')
                loss1 = 0
                for i in range(self.num_cls):
                    loss1 += bce2d(output[i], tag[i])
                loss2 = self.criterion(output2, reports_ids, reports_masks)
                loss = loss1 * 0.5 + loss2
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update()
                log = {'train_loss': train_loss / len(self.train_dataloader)}

        num_val = 2130 if self.args.dataset_name == 'mimic_cxr' else 296
        num_test = 3858 if self.args.dataset_name == 'mimic_cxr' else 590
        acc_val = [0.0 for i in range(self.num_cls)]

        sum_1_gt_val = [0.0 for i in range(self.num_cls)]
        sum_1_pre_val = [0.0 for i in range(self.num_cls)]
        sum_1_right_val = [0.0 for i in range(self.num_cls)]

        acc_test = [0.0 for i in range(self.num_cls)]

        sum_1_gt_test = [0.0 for i in range(self.num_cls)]
        sum_1_pre_test = [0.0 for i in range(self.num_cls)]
        sum_1_right_test = [0.0 for i in range(self.num_cls)]
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - test' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.test_dataloader):
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
                        pre[pre>=0.5] = 1
                        pre[pre<0.5] = 0
                        acc_test[i] += float(torch.sum(pre == tag[i]))
                        sum_1_gt_test[i] += float(torch.sum(tag[i]))
                        sum_1_pre_test[i] += float(torch.sum(pre))
                        sum_1_right_test[i] += float(torch.sum(pre * tag[i]))

                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[-1][:, 1:].cpu().numpy())
                    reports = [''] * len(reports_ids[-1])
                    for j in range(len(output2)):
                        tmp = self.model.tokenizer.decode_batch(output2[j].cpu().numpy())
                        for i in range(len(reports_ids[-1])):
                            if self.args.dataset_name == 'mimic_cxr' or check(ground_truths[i],j):
                                reports[i] = reports[i] + ' ' + tmp[i]
                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    pbar.update()
                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                score = test_met
                # import pdb;pdb.set_trace()
                for report,gt in zip(reports,ground_truths):
                    f2.write(report)
                    f2.write('\n')
                    f2.write(gt)
                    f2.write('\n')
                    f2.flush()
                f.write("\nbleu_1: %f ; bleu_2: %f ; bleu_3: %f ; bleu_4: %f ; METEOR: %f ; ROUGE: %f" % (
                    score['BLEU_1'], score['BLEU_2'], score['BLEU_3'], score['BLEU_4'], score['METEOR'],
                    score['ROUGE_L']))
                log.update(**{'val_' + k: v for k, v in test_met.items()})
                log.update(**{'test_' + k: v for k, v in test_met.items()})

        for i in range(self.num_cls):
            acc_test[i] /= num_test
        f.write("\ninformation of test: ")
        f.write("\nacc: ")
        f.write(output_func(acc_test))
        f.write("\nF1: ")
        F1_test = F1_score(sum_1_gt_test, sum_1_pre_test, sum_1_right_test,self.num_cls)
        f.write(output_func(F1_test))

        self.lr_scheduler.step()
        return log
    
class Trainer_Multi_Cls_ABL_two(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_one_dataloader, train_two_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer_Multi_Cls_ABL_two, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_one_dataloader = train_one_dataloader
        self.train_two_dataloader = train_two_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_cls = 7

    def _train_epoch(self, epoch):
        f = open('%s.txt' % self.args.save_dir, 'a+')
        f.write("\n+++++++++++++++++++++++++++++++++ epoch %d +++++++++++++++++++++++++++++++++" % epoch)
        train_loss = 0
        self.model.train()
        
        with tqdm(desc='Epoch %d - train_one' % epoch, unit='it', total=len(self.train_one_dataloader)) as pbar:
            for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.train_one_dataloader):
                images = images.to(self.device)
                for i in range(len(reports_ids)):
                    reports_ids[i] = reports_ids[i].to(self.device)
                    reports_masks[i] = reports_masks[i].to(self.device)

                for i in range(self.num_cls):
                    tag[i] = tag[i].to(self.device)
                    tag[i] = torch.squeeze(tag[i], dim=1)
                output, output2 = self.model(images, reports_ids, mode='train')
                loss1 = 0
                for i in range(self.num_cls):
                    loss1 += bce2d(output[i], tag[i])

                output2_one = []
                output2_one.append(output2[-1])
                reports_ids_one = []
                reports_ids_one.append(reports_ids[-1])
                reports_masks_one = []
                reports_masks_one.append(reports_masks[-1])

                loss2 = self.criterion(output2_one, reports_ids_one, reports_masks_one)
                loss = loss1 * 0.5 + loss2
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update()
            log = {'train_one_loss': train_loss / len(self.train_one_dataloader)}
        
        with tqdm(desc='Epoch %d - train_two' % epoch, unit='it', total=len(self.train_two_dataloader)) as pbar:
            for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.train_two_dataloader):
                images = images.to(self.device)
                for i in range(len(reports_ids)):
                    reports_ids[i] = reports_ids[i].to(self.device)
                    reports_masks[i] = reports_masks[i].to(self.device)

                for i in range(self.num_cls):
                    tag[i] = tag[i].to(self.device)
                    tag[i] = torch.squeeze(tag[i], dim=1)
                output, output2 = self.model(images, reports_ids, mode='train')
                loss1 = 0
                for i in range(self.num_cls):
                    loss1 += bce2d(output[i], tag[i])

                output2_two = []
                for i in range(len(output2)-1):
                    output2_two.append(output2[i])
                reports_ids_two = []
                for i in range(len(reports_ids)-1):
                    reports_ids_two.append(reports_ids[i])
                reports_masks_two = []
                for i in range(len(reports_masks)-1):
                    reports_masks_two.append(reports_masks[i])

                loss2 = self.criterion(output2_two, reports_ids_two, reports_masks_two)
                loss = loss1 * 0.5 + loss2
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update()
            log = {'train_two_loss': train_loss / len(self.train_two_dataloader)}

        num_val = 2130 if self.args.dataset_name == 'mimic_cxr' else 296
        num_test = 3858 if self.args.dataset_name == 'mimic_cxr' else 590
        acc_val = [0.0 for i in range(self.num_cls)]

        sum_1_gt_val = [0.0 for i in range(self.num_cls)]
        sum_1_pre_val = [0.0 for i in range(self.num_cls)]
        sum_1_right_val = [0.0 for i in range(self.num_cls)]

        acc_test = [0.0 for i in range(self.num_cls)]

        sum_1_gt_test = [0.0 for i in range(self.num_cls)]
        sum_1_pre_test = [0.0 for i in range(self.num_cls)]
        sum_1_right_test = [0.0 for i in range(self.num_cls)]

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - test' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                    images = images.to(self.device)
                    for i in range(len(reports_ids)):
                        reports_ids[i] = reports_ids[i].to(self.device)
                        reports_masks[i] = reports_masks[i].to(self.device)
                    for i in range(self.num_cls):
                        tag[i] = tag[i].to(self.device)
                        tag[i] = torch.squeeze(tag[i], dim=1)

                    output, output2 = self.model(images, mode='sample')

                    flag_list = [0] * len(reports_ids[-1])
                    for i in range(self.num_cls):
                        pre = torch.sigmoid(output[i].view(-1))
                        pre[pre>=0.5] = 1
                        pre[pre<0.5] = 0
                        tmp = pre.cpu().numpy()
                        for j in range(len(pre)):
                            if tmp[j] == 1:
                                flag_list[j] = 1
                        acc_test[i] += float(torch.sum(pre == tag[i]))
                        sum_1_gt_test[i] += float(torch.sum(tag[i]))
                        sum_1_pre_test[i] += float(torch.sum(pre))
                        sum_1_right_test[i] += float(torch.sum(pre * tag[i]))

                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[-1][:, 1:].cpu().numpy())
                    reports = [''] * len(reports_ids[-1])
                    for j in range(len(output2)):
                        tmp = self.model.tokenizer.decode_batch(output2[j].cpu().numpy())
                        for i in range(len(reports_ids[-1])):
                            if flag_list[i] == 0:
                                if j == len(output2)-1:
                                    reports[i] = tmp[i]
                            else:
                                if j < len(output2)-1:
                                    if self.args.dataset_name == 'mimic_cxr' or check(ground_truths[i],j):
                                        reports[i] = reports[i] + ' ' + tmp[i]

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    pbar.update()
                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                score = test_met
                f.write("\nbleu_1: %f ; bleu_2: %f ; bleu_3: %f ; bleu_4: %f ; METEOR: %f ; ROUGE: %f" % (
                    score['BLEU_1'], score['BLEU_2'], score['BLEU_3'], score['BLEU_4'], score['METEOR'],
                    score['ROUGE_L']))
                log.update(**{'val_' + k: v for k, v in test_met.items()})
                log.update(**{'test_' + k: v for k, v in test_met.items()})
                #with open('')
        for i in range(self.num_cls):
            acc_test[i] /= num_test
        f.write("\ninformation of test: ")
        f.write("\nacc: ")
        f.write(output_func(acc_test))
        f.write("\nF1: ")
        F1_test = F1_score(sum_1_gt_test, sum_1_pre_test, sum_1_right_test,self.num_cls)
        f.write(output_func(F1_test))

        self.lr_scheduler.step()
        return log

class Trainer_Multi_Cls_two(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_one_dataloader, train_two_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer_Multi_Cls_two, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_one_dataloader = train_one_dataloader
        self.train_two_dataloader = train_two_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_cls = 7

    def _train_epoch(self, epoch):
        f = open('%s.txt' % self.args.save_dir, 'a+')
        f.write("\n+++++++++++++++++++++++++++++++++ epoch %d +++++++++++++++++++++++++++++++++" % epoch)
        train_loss = 0
        self.model.train()
        
        with tqdm(desc='Epoch %d - train_one' % epoch, unit='it', total=len(self.train_one_dataloader)) as pbar:
            for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.train_one_dataloader):
                images = images.to(self.device)
                for i in range(len(reports_ids)):
                    reports_ids[i] = reports_ids[i].to(self.device)
                    reports_masks[i] = reports_masks[i].to(self.device)

                for i in range(self.num_cls):
                    tag[i] = tag[i].to(self.device)
                    tag[i] = torch.squeeze(tag[i], dim=1)
                output, output2 = self.model(images, reports_ids, mode='train')
                loss1 = 0
                for i in range(self.num_cls):
                    loss1 += bce2d(output[i], tag[i])

                output2_one = []
                output2_one.append(output2[-1])
                reports_ids_one = []
                reports_ids_one.append(reports_ids[-1])
                reports_masks_one = []
                reports_masks_one.append(reports_masks[-1])

                loss2 = self.criterion(output2_one, reports_ids_one, reports_masks_one)
                loss = loss1 * 0.5 + loss2
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update()
            log = {'train_one_loss': train_loss / len(self.train_one_dataloader)}
        
        with tqdm(desc='Epoch %d - train_two' % epoch, unit='it', total=len(self.train_two_dataloader)) as pbar:
            for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.train_two_dataloader):
                images = images.to(self.device)
                for i in range(len(reports_ids)):
                    reports_ids[i] = reports_ids[i].to(self.device)
                    reports_masks[i] = reports_masks[i].to(self.device)

                for i in range(self.num_cls):
                    tag[i] = tag[i].to(self.device)
                    tag[i] = torch.squeeze(tag[i], dim=1)
                output, output2 = self.model(images, reports_ids, mode='train')
                loss1 = 0
                for i in range(self.num_cls):
                    loss1 += bce2d(output[i], tag[i])

                output2_two = []
                for i in range(len(output2)-1):
                    output2_two.append(output2[i])
                reports_ids_two = []
                for i in range(len(reports_ids)-1):
                    reports_ids_two.append(reports_ids[i])
                reports_masks_two = []
                for i in range(len(reports_masks)-1):
                    reports_masks_two.append(reports_masks[i])

                loss2 = self.criterion(output2_two, reports_ids_two, reports_masks_two)
                loss = loss1 * 0.5 + loss2
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                pbar.set_postfix(loss=train_loss / (batch_idx + 1))
                pbar.update()
            log = {'train_two_loss': train_loss / len(self.train_two_dataloader)}

        num_val = 2130 if self.args.dataset_name == 'mimic_cxr' else 296
        num_test = 3858 if self.args.dataset_name == 'mimic_cxr' else 590
        acc_val = [0.0 for i in range(self.num_cls)]

        sum_1_gt_val = [0.0 for i in range(self.num_cls)]
        sum_1_pre_val = [0.0 for i in range(self.num_cls)]
        sum_1_right_val = [0.0 for i in range(self.num_cls)]

        acc_test = [0.0 for i in range(self.num_cls)]

        sum_1_gt_test = [0.0 for i in range(self.num_cls)]
        sum_1_pre_test = [0.0 for i in range(self.num_cls)]
        sum_1_right_test = [0.0 for i in range(self.num_cls)]

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            with tqdm(desc='Epoch %d - test' % epoch, unit='it', total=len(self.test_dataloader)) as pbar:
                for batch_idx, (images_id, images, tag, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                    images = images.to(self.device)
                    for i in range(len(reports_ids)):
                        reports_ids[i] = reports_ids[i].to(self.device)
                        reports_masks[i] = reports_masks[i].to(self.device)
                    for i in range(self.num_cls):
                        tag[i] = tag[i].to(self.device)
                        tag[i] = torch.squeeze(tag[i], dim=1)

                    output, output2 = self.model(images, mode='sample')

                    flag_list = [0] * len(reports_ids[-1])
                    for i in range(self.num_cls):
                        pre = torch.sigmoid(output[i].view(-1))
                        pre[pre>=0.5] = 1
                        pre[pre<0.5] = 0
                        tmp = pre.cpu().numpy()
                        for j in range(len(pre)):
                            if tmp[j] == 1:
                                flag_list[j] = 1
                        acc_test[i] += float(torch.sum(pre == tag[i]))
                        sum_1_gt_test[i] += float(torch.sum(tag[i]))
                        sum_1_pre_test[i] += float(torch.sum(pre))
                        sum_1_right_test[i] += float(torch.sum(pre * tag[i]))

                    ground_truths = self.model.tokenizer.decode_batch(reports_ids[-1][:, 1:].cpu().numpy())
                    reports = [''] * len(reports_ids[-1])
                    for j in range(len(output2)):
                        tmp = self.model.tokenizer.decode_batch(output2[j].cpu().numpy())
                        for i in range(len(reports_ids[-1])):
                            if flag_list[i] == 0:
                                if j == len(output2)-1:
                                    reports[i] = tmp[i]
                            else:
                                if j < len(output2)-1:
                                    if self.args.dataset_name == 'mimic_cxr' or check(ground_truths[i],j):
                                        reports[i] = reports[i] + ' ' + tmp[i]

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    pbar.update()
                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                score = test_met
                f.write("\nbleu_1: %f ; bleu_2: %f ; bleu_3: %f ; bleu_4: %f ; METEOR: %f ; ROUGE: %f" % (
                    score['BLEU_1'], score['BLEU_2'], score['BLEU_3'], score['BLEU_4'], score['METEOR'],
                    score['ROUGE_L']))
                log.update(**{'val_' + k: v for k, v in test_met.items()})
                log.update(**{'test_' + k: v for k, v in test_met.items()})

        for i in range(self.num_cls):
            acc_test[i] /= num_test
        f.write("\ninformation of test: ")
        f.write("\nacc: ")
        f.write(output_func(acc_test))
        f.write("\nF1: ")
        F1_test = F1_score(sum_1_gt_test, sum_1_pre_test, sum_1_right_test,self.num_cls)
        f.write(output_func(F1_test))

        self.lr_scheduler.step()
        return log