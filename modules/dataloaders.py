import torch
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader,DistributedSampler
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset,IuxrayMultiImageDataset_Multi_Cls,MimiccxrSingleImageDataset_Multi_Cls


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)

class Collate_fn():
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, data):
        images_id, images, tag, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)
        tag_list = []
        targets = []
        targets_masks = []

        t= np.arange(2)

        for _ in range(8):
            targets_ = np.zeros((len(images_id), 1), dtype=int)
            for i in range(len(images_id)):
                targets_[i, 0] = tag[i][_]
            tag_list.append(torch.LongTensor(targets_))

        for _ in range(len(seq_lengths[0])):
            max_seq_length = 0
            for i, seq_length in enumerate(seq_lengths):
                max_seq_length = max(max_seq_length, seq_length[_])

            target = np.zeros((len(reports_ids), max_seq_length), dtype=int)
            target_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)


            for i, report_ids in enumerate(reports_ids):
                target[i, :len(report_ids[_])] = report_ids[_]

            for i, report_masks in enumerate(reports_masks):
                #print(len(tag))
                if _+1 < len(seq_lengths[0]) and tag[i][_] == 0:
                    p_ = np.random.choice(a=t, size=1, replace=False, p=self.rate[_])
                
                    if p_ == 0:
                        report_masks[_] = report_masks[_] * 0
                 
                target_masks[i, :len(report_masks[_])] = report_masks[_]


            targets.append(torch.LongTensor(target))
            targets_masks.append(torch.FloatTensor(target_masks))

        return images_id, images, tag_list, targets, targets_masks

class R2DataLoader_Multi_Cls_ABL(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.p_list = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 1.000]
        self.rate = []

        for i in range(len(self.p_list)):
            self.rate.append([1-self.p_list[i], self.p_list[i]])

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset_Multi_Cls(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset_Multi_Cls(self.args, self.tokenizer, self.split, transform=self.transform)
        RANK = int(os.environ['SLURM_PROCID'])
        LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
        GPU_NUM = int(os.environ['SLURM_NTASKS'])
        IP = os.environ['SLURM_STEP_NODELIST']
        sampler = DistributedSampler(self.dataset,num_replicas=GPU_NUM, rank=RANK)
        
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            # 'shuffle': self.shuffle,
            'collate_fn': Collate_fn(self.rate),
            'num_workers': self.num_workers,
            'sampler':sampler
        }
        super().__init__(**self.init_kwargs)

class R2DataLoader_Multi_Cls(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset_Multi_Cls(self.args, self.tokenizer, self.split,
                                                            transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset_Multi_Cls(self.args, self.tokenizer, self.split,
                                                               transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, tag, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
        tag_list = []
        targets = []
        targets_masks = []
        for _ in range(8):
            targets_ = np.zeros((len(images_id), 1), dtype=int)
            for i in range(len(images_id)):
                targets_[i, 0] = tag[i][_]
            tag_list.append(torch.LongTensor(targets_))

        for _ in range(len(seq_lengths[0])):
            max_seq_length = 0
            for i, seq_length in enumerate(seq_lengths):
                max_seq_length = max(max_seq_length, seq_length[_])

            target = np.zeros((len(reports_ids), max_seq_length), dtype=int)
            target_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

            for i, report_ids in enumerate(reports_ids):
                target[i, :len(report_ids[_])] = report_ids[_]

            for i, report_masks in enumerate(reports_masks):
                target_masks[i, :len(report_masks[_])] = report_masks[_]

            targets.append(torch.LongTensor(target))
            targets_masks.append(torch.FloatTensor(target_masks))

        return images_id, images, tag_list, targets, targets_masks

class R2DataLoader_classify(DataLoader):
    def __init__(self, args, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = MimiccxrSingleImageDataset_Multi_Cls(self.args, self.tokenizer, self.split,
                                                            transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset_classify(self.args, self.split,
                                                               transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, tag = zip(*data)
        images = torch.stack(images, 0)
        tag_list = []
        for _ in range(14):
            targets_ = np.zeros((len(images_id), 1), dtype=int)
            for i in range(len(images_id)):
                targets_[i, 0] = tag[i][_]
            tag_list.append(torch.LongTensor(targets_))
        
        return images_id, images, tag_list