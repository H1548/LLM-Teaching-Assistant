# Transformer-Based Teachinng Assistant
This was my big final project for my Computer Science undergraduates course, which was, from scratch, pretraining and fine-tuning a transformer-based model to give a mark and feedback to an input which consisted of:
- question - e.g. "What is 2 + 2"
- marking criteria - e.g. "If student answers 4, give full marks"
- Total marks - e.g. "1"
The corresponding expected output of the fine-tuned transformer-based model should have consisted of: 
- marks - e.g. "1" if correct or "0" if incorrect
- feedback - e.g. "Your answer is incorrect try again, this time, ensure you have added correctly."

For more details on the entire project click this link to access the paper: https://drive.google.com/file/d/13F2R1bziUjhzPMcqMSaj0OMmZSi5X_Sl/view?usp=drive_link

However, The model did not perform well due to this being my fist time building a transformer model, the lack of technical knowledge of array/tensor manipulation, dimensions in arrays/tensors, etc, lead the model not to perform well and output poorly crafted reponses when prompted. Furthermore if one inspects the code for the transformer model will realize that there are prevlant bugs that need to be fixed that can potentially fix the model's performance. This will further be explained further down this readme page.

# Features
- Transformer-based model that can mark and give feedback to student answers according to the marking criteria given
- Transformer was pretrained and fine-tuned from scratch using **Pytorch**
- Encoder-Decoder architecture
- model can be prompted Via the prompting.py file

# Installation 
## Clone the repository 
```bash
git clone https://github.com/H1548/LLM-Teaching-Assistant.git
cd LLM-Teaching-Assistant
```
## Install Dependencies
```bash 
pip install -r requirements.txt
```
## pretrain transformer 
```bash 
python pretraintrainingloop.py
```
## Fine-tune the transformer 
load the correct checkpoint within the 'finetuningtrainloop.py' file before running the finetuning trainingloop file
```bash
python finetuningtrainloop.py
```
## evaluate model
once again after fine tuning, checkpoints should be stored in the 'fine-tuningcheckpoint' folder, load the correct checkpoint within the source code file 'Evaluate_LLM.py'
```bash 
python Evaluate_LLM.py
```
## Prompt the model
once again load the correct checkpoints, and run the prompting.py file, you will be met with a 'Prompt:', here you need to provide an input which consists of:
- The exam question
- the corresponding answer
- the relevant marking criteria 
- the total marks of the question
```bash 
python Prompting.py
Prompt: Prompt goes here....
```
# Results
As this was my first time evaluating the transformers ability to generate text, the execution of these tests may have not been executed well.

- Train-loss: 0.0482
- Val-loss: 0.0486
- Accuracy score: 0.0048
- BLEU Score: 3.6615
- METEOR Score: 0.001156

As shown in those results despite the really low train and val loss during fine-tuning the model performed very poor when its text generation was evaluated with the use of metrics like accuracy, BLEU Score and the METEOR Score.

# Project Structure
```text
.
|-- DataSet/      # folder stores datasets that are used for training
|-- EncDecCheckpoint   # folder stores model's pretraining parameters
|-- Fine-TuneCheckpoint     # folder that stores model's fine-tuning parameters
|-- SubPrograms         # Extra programming files that clean, extract and split data
|-- .gitignore          # ignore certain file types
|-- Evaluate_LLM.py        # File evaluates models text generation with metrics like BLEU, METEOR and Accuracy
|-- FineTuneDataLoader.py   # File loads and formats data into train and val sets
|-- finetuningtrainloop.py # File exectutes the training loop for finetuning
|-- modelfinetuning.py     # File contains code for the fine-tuning architecture of the transformer model
|-- pretrainmodel.py       # File contains code for the pre-training architecture of the transfrmer model
|-- Pretrainingtrainloop.py   # File exectutes the training loop for pretraining
|-- Prompting.py            # once run, you can prompt the model by submitting your question and answer
|-- README                  # Project Description 
|-- requirements.txt        # Project dependencies 
|-- TestsetLoader.py        # File loads and formats the testing data
|-- tokenizer.json          # contains a dictionary of tokens for each subword etc
|-- utils.py                # additional functions 
```

# Future Work
I am hoping to go through my code for this model again, identify the mistakes made and some bugs that stand out. Write a detailed analysis in a seperate 'Lessons Learned' section for this readme page. Furhtermore, I wil re-implement the model with a new learning objective for pretraining which will be span corruption from the T5 paper to see if i can improve the model and achieve it's core objective.