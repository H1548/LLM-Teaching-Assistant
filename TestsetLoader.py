
import json
from tokenizers import Tokenizer
from utils import pad_sequence 

tokenize = Tokenizer.from_file("tokenizer.json")

Sep_token = tokenize.token_to_id('[SEP]')
CLS_token = tokenize.token_to_id('[CLS]')
Pad_token = tokenize.token_to_id('[PAD]')

def load_test(data_path):
    with open(data_path, 'rb') as file:
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
            modified_output = pad_sequence()
            input_list.append(modified_input)
            output_list.append(modified_output)



    test_data = list(zip(input_list, output_list))
    return test_data
