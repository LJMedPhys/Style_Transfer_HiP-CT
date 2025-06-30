import torch
import json
import torch

class Config:
    def __init__(self, config_data):
        self.name_run = config_data['name_run']
        self.batch_size = config_data['batch_size']
        self.image_size = config_data['image_size']
        self.test_split = config_data['test_split']
        self.learning_rate_G = config_data['learning_rate_G']
        self.learning_rate_D = config_data['learning_rate_D']
        self.n_residual_blocks = config_data['n_residual_blocks']
        self.beta1 = config_data['beta1']
        self.beta2 = config_data['beta2']
        self.epochs = config_data['epochs']
        self.lambda_A = config_data['lambda_A']
        self.lambda_B = config_data['lambda_B']
        self.lambda_identity = config_data['lambda_identity']
        self.skip = config_data['skip']
        # Assuming the device field remains dynamically assigned
        self.device = torch.device(config_data["device"] if torch.cuda.is_available() else "cpu")
        self.path_checkpoints = config_data['path_checkpoints']
        self.paths_A_train = config_data['paths_A_train']
        self.paths_A_test = config_data['paths_A_test']
        self.paths_A_val = config_data['paths_A_val']
        self.paths_B_train = config_data['paths_B_train']
        self.paths_B_test = config_data['paths_B_test']
        self.paths_B_val = config_data['paths_B_val']
        

def load_config_from_json(json_file):
    with open(json_file, 'r') as f:
        config_data = json.load(f)
    return Config(config_data)