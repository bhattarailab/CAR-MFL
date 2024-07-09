import sys
import copy
import os
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from torch.nn.functional import softmax
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC

from datasets.mimic import MimicPublic
from .ClientTrainers import ClassificationTrainer as ClientClassificationTrainer, UnimodalClassificationTrainer, ClassificationTrainerRAG
from .ServerTrainers import ClassificationTrainer as ServerClassificationTrainer


from utils.retrieval import ModalityRetrival

class FedAvgIn:
    def __init__(self, args, wandb):
        self.args = args
        self.wandb = wandb
        self.num_mm_clients = args.num_clients ## Needed from Args
        self.total_comms = args.comm_rounds ## Needed from Args
        self.num_img_clients = args.img_clients
        self.num_txt_clients = args.txt_clients
        self.num_clients = self.num_mm_clients + self.num_img_clients + self.num_txt_clients

        self.setup_clients()
        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
            "AUCperLabel":  MultilabelAUROC(num_labels=14, average="none", thresholds=None)
        })
        self.val_track = []

    def setup_clients(self):
        self.server = ServerClassificationTrainer(self.args, self.args.server_config_path, False)
        self.clients_id = self.server.config.clients_config.arr
        mm_clients = [ClientClassificationTrainer(self.args, self.clients_id[i], self.args.client_config_path, False) for i in range(self.num_mm_clients)]
        img_clients = [UnimodalClassificationTrainer(self.args, self.clients_id[i+self.num_mm_clients], self.args.client_config_path, False, "image") for i in range(self.num_img_clients)]
        txt_clients = [UnimodalClassificationTrainer(self.args, self.clients_id[i+self.num_mm_clients + self.num_img_clients], self.args.client_config_path, False, "text") for i in range(self.num_txt_clients)]
        self.clients = mm_clients + img_clients + txt_clients

    def test(self):
        self.server.model.eval()
        self.server.model.cuda()
        with tqdm(self.server.test_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.server.model(self.server.tokenizer, images, text)
                self.evaluator.update(output["logits"], label.long())
        metrics = self.evaluator.compute()
        print(f"AUC : {metrics['AUC']}")
        print(f"AUCperLabel : {metrics['AUCperLabel']}")
        self.wandb.log({"Test AUC(Aggregrated)":metrics['AUC'].item()}, step=self.cur_comms)
        self.evaluator.reset()

    def dispatch(self):
        print("-------------Distributing Models blueprints to training centers------------------------")
        for client in self.clients:
            client.model.load_state_dict(self.server.model.state_dict(), strict=False) ## same dic_key value

    def aggregrate(self):
        global_dict = copy.deepcopy(self.server.model.state_dict())
        station_list = self.clients + [self.server]
        for k in global_dict.keys():
            if any(substring in k for substring in ["num_batches_tracked", "embeddings.position_ids"]):
                continue
            else:
                params = []
                weights = []
                for station in station_list:
                    para = station.model.state_dict().get(k)
                    if para is not None:
                        params.append(para)
                        weights.append(len(station.train_set))
                    total_weights = sum(weights)
                    normalized_weights = [w / total_weights for w in weights]
                weighted_params = [nw * para for nw, para in zip(normalized_weights, params)]
                weighted_sum = torch.sum(torch.stack(weighted_params, 0), dim=0)
                global_dict[k] = weighted_sum
        self.server.model.load_state_dict(global_dict)

    def val(self):
        self.server.model.eval()
        self.server.model.cuda()
        print('Validating Model:')
        with tqdm(self.server.val_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.server.model(self.server.tokenizer, images, text)
                self.evaluator.update(output["logits"], label.long())
        metrics = self.evaluator.compute()
        print(f"Val AUC : {metrics['AUC']}")
        self.wandb.log({"Val AUC(Aggregrated)":metrics['AUC'].item()}, step=self.cur_comms)
        self.evaluator.reset()
        return metrics['AUC'].item()

    def save_log(self):
        log_path = os.path.join(self.args.exp_dir, "server", "agg_val_aucs.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

    def run(self):
        self.cur_comms = 0
        self.val_auc = 0
        for comms in range(self.total_comms):
            print(f"-----------------------Communication Round: {comms}-------------------------------")
            self.dispatch()
            print(f"-----------------------Training Server Model in Server Data------------------------------")
            self.server.run(comms)
            print(f"-----------------------Training Client Models in Clients Data------------------------------")
            for client in self.clients:
                client.run(comms)
            print(f"-----------Performing global aggregration--------------------------")
            self.aggregrate()
            print("---------------Evaluating Aggregrated Model in Val Set-------------------------------")
            cur_auc = self.val()
            self.val_track.append(cur_auc)
            if cur_auc > self.val_auc:
                self.val_auc = cur_auc
                self.server.save_best(self.cur_comms)
            self.cur_comms +=1
            self.save_log()
            import gc
            gc.collect()
        self.server.load_best()
        self.test()

class FedAvgInRAG(FedAvgIn):
    def __init__(self, args, wandb):
        super(FedAvgInRAG, self).__init__(args, wandb)
        self.get_public_dset()
    
    def setup_clients(self):
        self.server = ServerClassificationTrainer(self.args, self.args.server_config_path, False)
        self.clients_id = self.server.config.clients_config.arr
        mm_clients = [ClientClassificationTrainer(self.args, self.clients_id[i], self.args.client_config_path, False) for i in range(self.num_mm_clients)]
        img_clients = [ClassificationTrainerRAG(self.args, self.clients_id[i+self.num_mm_clients], self.args.client_config_path, False, "image") for i in range(self.num_img_clients)]
        txt_clients = [ClassificationTrainerRAG(self.args, self.clients_id[i+self.num_mm_clients + self.num_img_clients], self.args.client_config_path, False, "text") for i in range(self.num_txt_clients)]
        self.clients = mm_clients + img_clients + txt_clients


    def get_public_dset(self):
        partition_path = f'partitions/{self.server.dset_name}_{self.server.config.dataset.view}_{self.server.config.dataset.partition}.pkl'
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)
        train_set = MimicPublic(self.server.config.dataset.img_path, self.server.config.dataset.ann_path, self.server.config.dataset.view, "eval")
        self.publid_idx = train_idx = data_partition["server"]
        self.public_dset = Subset(train_set, train_idx)
        self.public_dset_loader = DataLoader(self.public_dset,  batch_size=self.server.config.dataloader.eval_batch_size, shuffle=False, num_workers= self.server.config.dataloader.num_workers, pin_memory=True, drop_last=False)

            
    def aggregrate(self):
        global_dict = copy.deepcopy(self.server.model.state_dict())
        model_list = self.clients + [self.server]
        # weights = [1 for _ in model_list]
        weights = [len(station.train_set) for station in model_list]
        weights_sum = sum(weights)
        norm_weights = [w / weights_sum for w in weights]
        alpha = self.args.alpha  #contribution level for noisy
        img_agg = norm_weights.copy()
        txt_agg = norm_weights.copy()
        for i in range(self.num_img_clients):
            img_agg[self.num_mm_clients + i] = alpha * img_agg[self.num_mm_clients + i]
        for i in range(self.num_txt_clients):
            txt_agg[self.num_mm_clients + self.num_img_clients + i] = alpha * txt_agg[self.num_img_clients + self.num_mm_clients + i]
        noise_img_norm_weight = softmax(torch.tensor(img_agg).float(), dim=0).tolist() # noisy weights for image client
        noise_txt_norm_weight = softmax(torch.tensor(txt_agg).float(), dim=0).tolist() # noisy weights for text client


        for k in global_dict.keys():
            if any(substring in k for substring in ["num_batches_tracked", "embeddings.position_ids"]):
                continue
            else:
                if "text_encoder" in k:
                    mul_weights = noise_img_norm_weight #because image client have noisy text encoder
                elif "image_encoder" in k:
                    mul_weights = noise_txt_norm_weight #because text client hve noisy image encoder
                else:
                    mul_weights = norm_weights
                params = [nw * model.model.state_dict()[k]  for model, nw in zip(model_list, mul_weights)]
                weighted_sum = torch.sum(torch.stack(params, 0), dim=0)
                global_dict[k] = weighted_sum
        self.server.model.load_state_dict(global_dict)

    def run(self):
        self.cur_comms = 0
        self.val_auc = 0
        for comms in range(self.total_comms):
            print(f"-----------------------Communication Round: {comms}-------------------------------")
            self.setup_rag()
            self.dispatch()
            print(f"-----------------------Training Server Model in Server Data------------------------------")
            self.server.run(comms)
            print(f"-----------------------Training Client Models in Clients Data------------------------------")
            for i in range(self.num_mm_clients):
                self.clients[i].run(comms)
            for i in range(self.num_mm_clients, self.num_clients):
                self.clients[i].run(comms, self.img_vec, self.txt_vec, self.labels, self.idxs)
            print(f"-----------Performing global aggregration--------------------------")
            self.aggregrate()
            print("---------------Evaluating Aggregrated Model in Val Set-------------------------------")
            cur_auc = self.val()
            self.val_track.append(cur_auc)
            if cur_auc > self.val_auc:
                self.val_auc = cur_auc
                self.server.save_best(self.cur_comms)
            self.cur_comms +=1
            self.save_log()
            import gc
            gc.collect()
        self.server.load_best()
        self.test()
    
    def setup_rag(self):
        img_vec = []
        txt_vec = []
        idxs = []
        labels = []
        self.server.model.cuda()
        self.server.model.eval()
        print("---Genearting global data features ----------")
        with tqdm(self.public_dset_loader, unit="batch") as tepoch:
            for frame, gt, report, idx in tepoch:
                frame = frame.cuda()
                with torch.no_grad():
                    output = self.server.model(self.server.tokenizer, frame, report)
                img_vec.extend(output["image_features"].cpu().numpy())
                txt_vec.extend(output["caption_features"].cpu().numpy())
                idxs.extend(idx)
                labels.extend(gt)
        self.server.model.cpu()
        print(f"Total {len(img_vec)} features generated")
        self.img_vec = img_vec
        self.txt_vec = txt_vec
        self.idxs = idxs
        self.labels = labels

class FedAvgNoPublic(FedAvgIn):
    def __init__(self, args, wandb):
        super().__init__(args, wandb)
    

    def aggregrate(self):
        global_dict = copy.deepcopy(self.server.model.state_dict())
        station_list = self.clients
        global_dict = copy.deepcopy(station_list[0].model.state_dict())
        for k in global_dict.keys():
            if any(substring in k for substring in ["num_batches_tracked", "embeddings.position_ids"]):
                continue
            else:
                params = []
                weights = []
                for station in station_list:
                    para = station.model.state_dict().get(k)
                    if para is not None:
                        params.append(para)
                        weights.append(len(station.train_set))
                    total_weights = sum(weights)
                    normalized_weights = [w / total_weights for w in weights]
                weighted_params = [nw * para for nw, para in zip(normalized_weights, params)]
                weighted_sum = torch.sum(torch.stack(weighted_params, 0), dim=0)
                global_dict[k] = weighted_sum
        self.server.model.load_state_dict(global_dict)
    
    def run(self):
        self.cur_comms = 0
        self.val_auc = 0
        for comms in range(self.total_comms):
            print(f"-----------------------Communication Round: {comms}-------------------------------")
            self.dispatch()
            print(f"-----------------------Training Client Models in Clients Data------------------------------")
            for client in self.clients:
                client.run(comms)
            print(f"-----------Performing global aggregration--------------------------")
            self.aggregrate()
            print("---------------Evaluating Aggregrated Model in Val Set-------------------------------")
            cur_auc = self.val()
            self.val_track.append(cur_auc)
            if cur_auc > self.val_auc:
                self.val_auc = cur_auc
                self.server.save_best(self.cur_comms)
            self.cur_comms +=1
            self.save_log()
            import gc
            gc.collect()
        self.server.load_best()
        self.test()