import pickle
import re
import torch
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ').replace('\r', '') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

class IUXrayMultiModal(Dataset):
    def __init__(self, data_root, ann_root, view_type="frontal", split="train"):
        super().__init__()
        assert view_type in ["lateral", "frontal"]
        annFile = os.path.join(ann_root, f'iuxray_{view_type}.pkl')
        with open(annFile, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)
        self.data = loaded_data[split]
        if split=="train":
            self.transforms = transforms.Compose([
                    transforms.Resize((224, 224)),  # Resize the image to a fixed size (e.g., 224x224)
                    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for data augmentation
                    transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
                    transforms.ToTensor(),  # Convert the image to a PyTorch tensor (0-1 range)
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image using precomputed values
                ])
        else:
            self.transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])   
        self.image_root = data_root
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data_item = self.data[idx]
        report = data_item['report']

        relative_path = data_item["image"]
        img_path = os.path.join(self.image_root, relative_path)
        
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        label = data_item["label"]
        cleaned_report = clean_report_mimic_cxr(report)
        return image, torch.tensor(label), cleaned_report, idx
    
class IUXrayPublic(Dataset):
    def __init__(self, data_root, ann_root, view_type="frontal", dst_type="train"):
        super().__init__()
        assert view_type in ["lateral", "frontal"]
        annFile = os.path.join(ann_root, f'iuxray_{view_type}.pkl')
        with open(annFile, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)
        self.data = loaded_data["train"]
        if dst_type=="train":
            self.transforms = transforms.Compose([
                    transforms.Resize((224, 224)),  # Resize the image to a fixed size (e.g., 224x224)
                    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for data augmentation
                    transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
                    transforms.ToTensor(),  # Convert the image to a PyTorch tensor (0-1 range)
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image using precomputed values
                ])
        else:
            self.transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])   
        self.image_root = data_root
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data_item = self.data[idx]
        report = data_item['report']

        relative_path = data_item["image"]
        img_path = os.path.join(self.image_root, relative_path)
        
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        label = data_item["label"]
        cleaned_report = clean_report_mimic_cxr(report)
        return image, torch.tensor(label), cleaned_report, idx