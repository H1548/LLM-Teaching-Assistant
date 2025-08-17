import torch
import json
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenize = Tokenizer.from_file("tokenizer.json")
Sep_token = tokenize.token_to_id('[SEP]')
CLS_token = tokenize.token_to_id('[CLS]')



def load_data(dataset_path):
    with open(dataset_path, 'rb') as file:
        text = json.load(file)
    input_list = []
    output_list = []

    for item in text:
        question = item.get("question", "")
        marking_criteria = item.get("Marking Criteria", "")
        total_marks = item.get("Total marks", "")
        for answer in item.get("answers", []):
            student_answer = answer.get("answer", "")
            marks = answer.get("marks", "")
            feedback = answer.get("feedback", "")
            
            combined_input = f"Question: {question} Marking Criteria: {marking_criteria} Total Marks: {total_marks} Student Answer: {student_answer}"
            combined_output = f"Marks: {marks} Feedback: {feedback}"
            
            # Tokenize input and output
            tok_input = tokenize.encode(combined_input)
            tok_output = tokenize.encode(combined_output)
            
            # Prepend [CLS] token and append [SEP] token for input
            modified_input = tok_input.ids
            # Prepend [CLS] token and append [SEP] token for output
            modified_output = [CLS_token] + tok_output.ids + [Sep_token]
            
            input_list.append(modified_input)
            output_list.append(modified_output)

    paired_data = list(zip(input_list, output_list))
    random.shuffle(paired_data)
    n = int(0.8 * len(paired_data))
    train_data = paired_data[:n]
    val_data = paired_data[n:]
    return train_data, val_data
