import torch
from modelfinetuning import Transformer


n_layer = 7

device = 'cuda' if torch.cuda.is_available() else 'cpu'






model = Transformer(n_layer)

parameters = torch.load('Fine-TuneCheckpoint\checkpoint_iter_10000.pt')
model.load_state_dict(parameters['model_state_dict'])

model = model.to(device)
    
while True:
    input_ids= input("Prompt: \n")
    generated_sequence =  model.generate_response(input_ids)
    print(f"Generated Sequence : {generated_sequence}")