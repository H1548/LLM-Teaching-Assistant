# Transformer-Based Teachinng Assistant
This was my big final project for my Computer Science undergraduates course, which was, from scratch, pretraining and fine-tuning a transformer-based model to give a mark and feedback to an input which consisted of:
- question - e.g. "What is 2 + 2"
- marking criteria - e.g. "If student answers 4, give full marks"
- Total marks - e.g. "1"
The corresponding expected output of the fine-tuned transformer-based model should have consisted of: 
- marks - e.g. "1" if correct or "0" if incorrect
- feedback - e.g. "Your answer is incorrect try again, this time, ensure you have added correctly."

For more details on the entire project click this link to access the paper: https://drive.google.com/file/d/13F2R1bziUjhzPMcqMSaj0OMmZSi5X_Sl/view?usp=drive_link

# Features
- Transformer-based model that can mark and give feedback to student answers according to the marking criteria given
- Transformer was pretrained and fine-tuned from scratch using **Pytorch**