

import sys
import os
import pickle
import copy
import operator

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append("..")

from utils.utils import find_closest_vector, find_top_k_closest_vectors, jaccard_similarity
from utils.retrieval import ModalityRetrival
from utils.config import parse_config
from datasets.mimic import MimicMultiModal, MimicPublic
from networks import get_mmclf, get_tokenizer, get_clf
from networks.optimizers import get_optimizer
from losses import get_criterion
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC




class ClassificationTrainer:
    def __init__(self, args, client_id, config_path, wandb=False):
        self.args = args
        self.client_id = client_id
        self.wandb = wandb
        self.config = parse_config(config_path)
        self.dset_name = self.config.dataset.dset_name
        self.load_data()
        self.load_model()

        self.val_track = []

        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
        })
        self.local_epoch = 0
        self.save_dir = os.path.join(self.args.exp_dir, f"client_{self.client_id}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        

    def load_data(self, use_public=False):

        partition_path = f'partitions/{self.dset_name}_{self.config.dataset.view}_{self.config.dataset.partition}.pkl'
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)
        
        train_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "train")
        client_partition = data_partition["client"]
        train_idx = client_partition[self.client_id]["train"]
        if use_public:
            public_train = data_partition["server"]
            train_idx += public_train
        val_idx = client_partition[self.client_id]["val"]
        self.train_set = Subset(train_set, train_idx)
        self.val_set = Subset(train_set, val_idx)

        self.train_loader = DataLoader(self.train_set, batch_size=self.config.dataloader.batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.val_loader = DataLoader(self.val_set, batch_size=self.config.dataloader.eval_batch_size, shuffle=False, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        print("------------------------------Data Loaded Successfully-------------------------")


    def load_model(self):
        self.model = get_mmclf(config=self.config.model)
        self.criterion = get_criterion(self.config.criterion.name, self.config.criterion)
        self.optimizer = get_optimizer(self.config.optimizer.name, self.model.parameters(), self.config.optimizer)
        self.tokenizer = get_tokenizer(self.config.model)
        # self.lr_scheduler = get_lr_scheduler(self.config.lr_scheduler.name, self.optimizer, self.config.lr_scheduler)
        self.grad_scaler =  torch.cuda.amp.GradScaler()
        print("------------------------------Model Loaded Successfully-------------------------")


    def run(self, comms):
        self.model.cuda()
        for i in range(self.config.train.local_epoch):
            print(f"Client_id:{self.client_id}  local_epoch:{self.local_epoch} communication round: {comms} round_epoch: {i}")
            self.train_epoch()
            self.local_epoch +=1
        self.model.cpu()
        import gc
        gc.collect()

    def train_epoch(self):
        self.model.train()
        print("Training Model:")
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, text)
                    loss = self.criterion(output["logits"], label)
                
                self.grad_scaler.scale(loss).backward()
                
                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
                
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())


class UnimodalClassificationTrainer:
    def __init__(self, args, client_id, config_path, wandb, modality):
        self.args = args
        self.client_id = client_id
        self.wandb = wandb
        self.config = parse_config(config_path)
        self.dset_name = self.config.dataset.dset_name
        self.modality = modality
        self.load_data()
        self.load_model()

        self.val_track = []

        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
        })
        self.local_epoch = 0
        self.save_dir = os.path.join(self.args.exp_dir, f"{self.modality}_client_{self.client_id}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        

    def load_data(self, use_public=False):

        partition_path = f'partitions/{self.dset_name}_{self.config.dataset.view}_{self.config.dataset.partition}.pkl'
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)
        
        train_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "train")
        client_partition = data_partition["client"]
        train_idx = client_partition[self.client_id]["train"]
        if use_public:
            public_train = data_partition["server"]
            train_idx += public_train
        val_idx = client_partition[self.client_id]["val"]
        self.train_set = Subset(train_set, train_idx)
        self.val_set = Subset(train_set, val_idx)

        self.train_loader = DataLoader(self.train_set, batch_size=self.config.dataloader.batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.val_loader = DataLoader(self.val_set, batch_size=self.config.dataloader.eval_batch_size, shuffle=False, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        print("------------------------------Data Loaded Successfully-------------------------")


    def load_model(self):
        self.model = get_clf(config=self.config.model, modality=self.modality)
        self.criterion = get_criterion(self.config.criterion.name, self.config.criterion)
        self.optimizer = get_optimizer(self.config.optimizer.name, self.model.parameters(), self.config.optimizer)
        # if self.modality == 'text':
        self.tokenizer = get_tokenizer(self.config.model)
        # self.lr_scheduler = get_lr_scheduler(self.config.lr_scheduler.name, self.optimizer, self.config.lr_scheduler)
        self.grad_scaler =  torch.cuda.amp.GradScaler()
        print("------------------------------Model Loaded Successfully-------------------------")

    def run(self, comms):
        self.model.cuda()
        for i in range(self.config.train.local_epoch):
            print(f"Client_id:{self.client_id}  local_epoch:{self.local_epoch} communication round: {comms} round_epoch: {i}")
            self.train_epoch()
            self.local_epoch +=1
        self.model.cpu()
        import gc
        gc.collect()

    def train_epoch(self):
        self.model.train()
        print("Training Model:")
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if self.modality == 'text':
                        output = self.model(self.tokenizer, text)
                    elif self.modality == 'image':
                        output = self.model(images)
                    loss = self.criterion(output["logits"], label)
                
                self.grad_scaler.scale(loss).backward()
                
                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
                
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())

    
class ClassificationTrainerRAG(ClassificationTrainer):
    def __init__(self, args, client_id, config_path, wandb, modality):
        super(ClassificationTrainerRAG, self).__init__(args, client_id, config_path, wandb)
        self.modality = modality
        self.save_dir = os.path.join(self.args.exp_dir, f"{self.modality}_client_{self.client_id}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.setup_rag()

    def setup_rag(self):
        # self.central_model = copy.deepcopy(self.model)
        partition_path = f'partitions/{self.dset_name}_{self.config.dataset.view}_{self.config.dataset.partition}.pkl'
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)
        train_set = MimicPublic(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "train")
        self.publid_idx = train_idx = data_partition["server"]
        self.public_train_dset = train_set
        
        eval_set = MimicPublic(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "eval")
        client_partition = data_partition["client"]
        self.local_train_idx = client_partition[self.client_id]["train"]
        self.local_train_dset = Subset(eval_set, self.local_train_idx)
        self.local_train_dsetloader = DataLoader(self.local_train_dset, batch_size=self.config.dataloader.eval_batch_size, shuffle=False, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        

    def generate_RAG_mapping(self, global_vec, global_labels):
        self.model.eval()
        self.model.cuda()
        local_img_vec = []
        local_txt_vec = []
        local_idx = []
        local_label = []
        print("Retriving Top K datasets")
        with tqdm(self.local_train_dsetloader, unit="batch") as tepoch:
            for frame, gt, report, idx in tepoch:
                frame = frame.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, frame, report)
                local_img_vec.extend(output["image_features"].cpu().numpy())
                local_txt_vec.extend(output["caption_features"].cpu().numpy())
                local_idx.extend(idx.numpy())
                local_label.extend(gt)
        if self.modality == "text":
            if self.args.use_refinement:
                print("----------Label Refining-------------")
                top_k_closet_idx, _ = find_top_k_closest_vectors(local_txt_vec, global_vec, 10)
            else:
                top_k_closet_idx, _ = find_closest_vector(local_txt_vec, global_vec)
        elif self.modality == "image":
            if self.args.use_refinement:
                print("----------Label Refining-------------")
                top_k_closet_idx, _ = find_top_k_closest_vectors(local_img_vec, global_vec, 10)
            else:
                top_k_closet_idx, _ = find_closest_vector(local_img_vec, global_vec)
        if self.args.use_refinement:
            closet_idx = []
            print("Refining the retrived data")
            for i in tqdm(range(len(top_k_closet_idx))):
                cur_label = local_label[i]
                top_ks = top_k_closet_idx[i]
                global_label = operator.itemgetter(*top_ks)(global_labels)
                similarities = [jaccard_similarity(cur_label.numpy(), label.numpy()) for label in global_label]
                closest_label_index = torch.argmax(torch.tensor(similarities)).item()
                closet_idx.append(top_ks[closest_label_index])
        else:
            closet_idx = top_k_closet_idx
        self.mappings = {a:b for (a,b) in zip(local_idx, closet_idx)}
        self.model.cpu()


    def retrive_data(self, local_idxs, global_idxs):
        map_idx = operator.itemgetter(*local_idxs)(self.mappings)
        global_idx = operator.itemgetter(*map_idx)(global_idxs)
        retrived_data = []
        for id in global_idx:
            image, _, text, _ = self.public_train_dset[id]
            if self.modality == "image":
                retrived_data.append(text)     
            elif self.modality == "text":
                retrived_data.append(image)
        if self.modality == "text":
            retrived_data = torch.stack(retrived_data, dim=0)
        return retrived_data
    
    def run(self, comms, img_vec, txt_vec, labels, idxs):
        print(f"client_id : {self.client_id} training started")
        if self.modality == "image":
            self.generate_RAG_mapping(img_vec, labels)
        elif self.modality == "text":
            self.generate_RAG_mapping(txt_vec, labels)
        for i in range(self.config.train.local_epoch):
            print(f"Client_id: {self.client_id} local_epoch{self.local_epoch} communication round: {comms} round_epoch: {i}")
            self.train_epoch(idxs)
            self.local_epoch += 1
            self.save_mappings(comms, idxs)
        self.model.cpu()
        import gc
        gc.collect()      
        

    def train_epoch(self, global_idxs):
        self.model.train()
        self.model.cuda()
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for frames, label, text, idx in tepoch:
                idx = idx.tolist()
                missed_data = self.retrive_data(idx, global_idxs)
                self.optimizer.zero_grad()
                label = label.cuda()
                if self.modality == 'text':
                    images = missed_data.to(frames.dtype).cuda()
                    report = text
                elif self.modality == 'image':
                    images = frames.cuda()
                    report = missed_data
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, report)
                    loss = self.criterion(output["logits"], label)
                
                self.grad_scaler.scale(loss).backward()
                
                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
                
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())
        

    def save_mappings(self, comms, global_index):
        mappings_path = os.path.join(self.save_dir, f"mappings_{comms}.pkl")
        dump_object = {"mappings":self.mappings , "global_idxs":global_index}
        with open(mappings_path, "wb") as f:
            pickle.dump(dump_object, f)
