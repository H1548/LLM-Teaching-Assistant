import mmap
from tqdm import tqdm
import os
import re

def clean_text(text):
    text = re.sub(r'[\x00]', '', text)  
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

def concatenate_files_in_chunks(file_paths, output_path, chunk_size=1024*1024):
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for file_path in file_paths:
            total_size = os.path.getsize(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for _ in tqdm(range(0, total_size, chunk_size), desc=f"Processing {file_path}"):
                    content = f.read(chunk_size)
                    if not content:
                        break
                    cleaned_content = clean_text(content)
                    out_file.write(cleaned_content) 

train_file_path = "train_split.txt"
val_file_path = "val_split.txt"
concatenated_file_path = "concatenated_openwebtext.txt"
concatenate_files_in_chunks([train_file_path, val_file_path], concatenated_file_path)