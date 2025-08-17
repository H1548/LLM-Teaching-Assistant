import torch
import json
from TestsetLoader import load_set
from modelfinetuning import Transformer
from utils import evaluate_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CUDA_LAUNCH_BLOCKING=1
print(device)

data_path = 'Test_dataset.json'
n_layer = 7

test_data = load_set(data_path)

model = Transformer(n_layer)

parameters = torch.load('Fine-TuneCheckpoint\checkpoint_iter_10000.pt')
model.load_state_dict(parameters['model_state_dict'])
 

model = model.to(device)


accuracy, bleu, meteor = evaluate_model(model, test_data)
print(f"Accuracy: {accuracy}\nBLEU Score: {bleu}\nMETEOR Score: {meteor}")