import os
import copy
import torch
import pickle
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC


from .ClientTrainers import ClassificationTrainer as ClientClassificationTrainer
from .ServerTrainers import ClassificationTrainer as ServerClassificationTrainer
class FedAvg:
    def __init__(self, args, wandb):
        self.args = args
        self.wandb = wandb
        self.num_clients = args.num_clients ## Needed from Args
        self.total_comms = args.comm_rounds ## Needed from Args
        
        self.clients = [ClientClassificationTrainer(args, i, self.args.client_config_path, False) for i in range(self.num_clients)]
        self.server = ServerClassificationTrainer(args, self.args.server_config_path, False)
        
        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
            "AUCperLabel":  MultilabelAUROC(num_labels=14, average="none", thresholds=None)
        })
        self.val_track = []
    
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
            client.model.load_state_dict(self.server.model.state_dict())
    
    def aggregrate(self):
        global_dict = copy.deepcopy(self.server.model.state_dict())
        model_list = self.clients + [self.server]
        weights = [len(station.train_set) for station in model_list]
        total_weights = sum(weights)
        normalized_weights = [w / total_weights for w in weights]
        for k in global_dict.keys():
            if any(substring in k for substring in ["num_batches_tracked", "embeddings.position_ids"]):
                continue
            else:
                params = [nw * model.model.state_dict()[k]  for model, nw in zip(model_list, normalized_weights)]
                weighted_sum = torch.sum(torch.stack(params, 0), dim=0)
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
