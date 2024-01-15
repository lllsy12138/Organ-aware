import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'],self.max_seq_length)[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])#classification token

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

#以下是新增部分
class BaseDataset_Multi_Cls(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        keyword_list = ['heart', 'lung', 'bone', 'pleural', 'airspace', 'thoracic_aorta', 'trachea',
                        'report'] if args.dataset_name == 'iu_xray' else ['others', 'trachea', 'heart', 'lung', 'pleural', 'thoracic_aorta', 'airspace', 
                        'bone', 'report']

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = []
            self.examples[i]['mask'] = []
            self.examples[i]['len'] = []
            for key in keyword_list:
                max_length = 60 if args.dataset_name == 'iu_xray' else 100
                tmp = tokenizer(self.examples[i][key], self.max_seq_length if key is not 'report' else max_length)
                self.examples[i]['ids'].append(tmp)
                self.examples[i]['mask'].append([1] * len(tmp))
                self.examples[i]['len'].append(len(tmp))
        self.examples = self.ann[self.split]

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset_Multi_Cls(BaseDataset_Multi_Cls):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        tag = example['tag']
        report_ids = example['ids']
        report_masks = example['mask']#mask的意义？？
        seq_length = example['len']
        sample = (image_id, image, tag, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset_Multi_Cls(BaseDataset_Multi_Cls):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        tag = example['tag']
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = example['len']
        sample = (image_id, image, tag, report_ids, report_masks, seq_length)
        return sample


class IuxrayMultiImageDataset_new_loss(BaseDataset_Multi_Cls):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = example['len']
        tag = example['tag']
        sample = (image_id, image, report_ids, report_masks, seq_length, tag)
        return sample


class MimiccxrSingleImageDataset_new_loss(BaseDataset_Multi_Cls):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = example['len']
        tag = example['tag']
        sample = (image_id, image, report_ids, report_masks, seq_length, tag)
        return sample


class BaseDataset_classify(Dataset):
    def __init__(self, args, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.split = split
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]

    def __len__(self):
        return len(self.examples)


class MimiccxrSingleImageDataset_classify(BaseDataset_classify):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        tag = example['tag']
        sample = (image_id, image, tag)
        return sample