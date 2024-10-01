import os
import random
import numpy as np
import torch

import argparse

from algorithms.ServerTrainers import ClassificationTrainer
from algorithms.FedAvg import FedAvg
from algorithms.FedAvgIn import FedAvgIn, FedAvgInRAG, FedAvgNoPublic

parser = argparse.ArgumentParser(description='Federated Learning')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    #To let the cuDNN use the same convolution every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_wandb(args):
    import wandb
    name = f"{str(args.name)}"

    wandb.init(
        project="qualitative",
        name = name,
        resume = None,
        config=args
    )

    return wandb

def args():
    parser.add_argument('--name', type=str, default='Test', help='The name for different experimental runs.')
    parser.add_argument('--exp_dir', type=str, default='./experiments/',
                        help='Locations to save different experimental runs.')
    parser.add_argument('--server_config_path', type=str, default='configs/server_configs.yaml',
                        help='Location for server configs')
    parser.add_argument('--client_config_path', type=str, default='configs/client_configs.yaml',
                        help='Location for client configs')
    parser.add_argument('--comm_rounds', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algorithm', type=str, default='standalone', choices=['standalone', 'full', 'fedavg', 'fedavgln', 'fedavgRAG', 'fedavgzeropublic'],
                        help='Choice of Federated Averages')
    parser.add_argument('--num_clients', type=int, default=10,
                        help='total number of multimodal clients')
    parser.add_argument('--img_clients', type=int, default=10,
                        help='total number of image clients')
    parser.add_argument('--txt_clients', type=int, default=10,
                        help='total number of text clients')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='noise_level')
    parser.add_argument('--save_clients', action="store_true", default=False)
    parser.add_argument('--use_refinement', action="store_true", default=False)

    
args()
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    wandb = init_wandb(args)
    if args.algorithm == 'standalone':
        trainer = ClassificationTrainer(args, args.server_config_path, wandb)
        trainer.run_standalone()
    elif args.algorithm == 'fedavg':
        engine = FedAvg(args, wandb)
        engine.run()
    elif args.algorithm == 'fedavgln':
        engine = FedAvgIn(args, wandb)
        engine.run()
    elif args.algorithm == 'fedavgRAG':
        engine = FedAvgInRAG(args, wandb)
        engine.run()
    elif args.algorithm == 'fedavgzeropublic':
        engine = FedAvgNoPublic(args, wandb)
        engine.run()
    else:
        raise ValueError(f"Not implemented {args.algorithm}")