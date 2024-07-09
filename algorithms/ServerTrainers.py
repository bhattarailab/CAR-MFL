

import sys
import os
import pickle

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append("..")

from utils.config import parse_config
from datasets.mimic import MimicMultiModal
from datasets.iu_xray import IUXrayMultiModal
from networks import get_mmclf, get_tokenizer
from networks.optimizers import get_optimizer
from losses import get_criterion
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC




class ClassificationTrainer:
    def __init__(self, args, config_path, wandb=False):
        self.args = args
        self.wandb = wandb

        self.config = parse_config(config_path)
        self.dset_name = self.config.dataset.dset_name
        self.load_data()
        self.load_model()

        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
        })
        self.cur_epoch = 0
        self.save_dir = os.path.join(self.args.exp_dir, "server")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.val_track = []

    def load_data(self):
        if self.dset_name == "mimic-cxr":
            partition_path = f'partitions/{self.dset_name}_{self.config.dataset.view}_{self.config.dataset.partition}.pkl'
            with open(partition_path, "rb") as f:
                data_partition = pickle.load(f)
            train_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "train")
            train_idx = data_partition["server"]
            self.train_set = Subset(train_set, train_idx)
            self.val_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "val")
            self.test_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "test")
        elif self.dset_name == "iuxray":
            self.train_set = IUXrayMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "train")
            self.val_set = IUXrayMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "val")
            self.test_set = IUXrayMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "test")

        self.train_loader = DataLoader(self.train_set, batch_size=self.config.dataloader.batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.val_loader = DataLoader(self.val_set, batch_size=self.config.dataloader.eval_batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.config.dataloader.eval_batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        print("------------------------------Data Loaded Successfully-------------------------")

    def load_model(self):
        self.model = get_mmclf(config=self.config.model)
        self.tokenizer = get_tokenizer(config=self.config.model)
        self.criterion = get_criterion(self.config.criterion.name, self.config.criterion)
        self.optimizer = get_optimizer(self.config.optimizer.name, self.model.parameters(), self.config.optimizer)
        self.grad_scaler =  torch.cuda.amp.GradScaler()
        print("------------------------------Model Loaded Successfully-------------------------")

    def save_best(self, comms):
        ckpt_path = os.path.join(self.save_dir, f"model_best.pth")
        torch.save({"net":self.model.state_dict(), "comms":comms}, ckpt_path)

    def load_best(self):
        ckpt_path = os.path.join(self.save_dir, f"model_best.pth")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["net"])
        print(f"Best Model is at comms : {checkpoint['comms']}")

    def save_log(self):
        log_path = os.path.join(self.save_dir, "val_aucs.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

    def run_standalone(self):
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        print("----------------Standalone Training-------------------------")
        self.val_auc = 0
        self.model.cuda()
        for i in range(self.config.train.total_epoch):
            print(f"Server: Epoch {i}")
            self.train_epoch()
            cur_auc = self.val()
            self.val_track.append(cur_auc)
            if cur_auc > self.val_auc:
                self.val_auc = cur_auc
                self.save_best(i)
            self.cur_epoch+=1
        print("------------------------------------------------------------")
        self.save_log()
        self.load_best()
        self.test()
    

    def test(self):
        self.model.cuda()
        self.model.eval()
        test_evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
            "AUCperLabel":  MultilabelAUROC(num_labels=14, average="none", thresholds=None)
        })
        with tqdm(self.test_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                test_evaluator.update(output["logits"], label.long())
        metrics = test_evaluator.compute()
        print(f"AUC : {metrics['AUC']}")
        print(f"AUCperLabel : {metrics['AUCperLabel']}")
        self.wandb.log({"Test AUC(Aggregrated)":metrics['AUC'].item()})
        self.evaluator.reset()


    def run(self, comms):
        self.model.cuda()
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        for i in range(self.config.train.local_epoch):
            print(f"Server:-  Comm round{comms} local_epoch:{self.cur_epoch}  round_epoch: {i}")
            self.train_epoch()
            self.cur_epoch +=1
            print("------------------------------------------------------------")
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

    def val(self):
        self.model.eval()
        print('Validating Model:')
        with tqdm(self.val_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                self.evaluator.update(output["logits"], label.long())
        metrics = self.evaluator.compute()
        print(f"Val AUC : {metrics['AUC']}")
        if self.wandb:
            self.wandb.log({"Val AUC(Server)":metrics['AUC'].item()}, step=self.cur_epoch)
        self.evaluator.reset()
        return metrics['AUC'].item()


        