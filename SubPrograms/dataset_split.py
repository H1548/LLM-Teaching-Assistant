import random

def split_dataset(input_file_path, train_file_path, val_file_path, val_percentage=0.10):
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(train_file_path, 'w', encoding='utf-8') as train_file, \
         open(val_file_path, 'w', encoding='utf-8') as val_file:
        
         for line in input_file:
            if random.random() < val_percentage:
                val_file.write(line)
            else:
                train_file.write(line)

input_file_path = "concatenated_openwebtext.txt"
train_file_path = "train_split.txt"
val_file_path = "val_split.txt"

split_dataset(input_file_path, train_file_path, val_file_path)