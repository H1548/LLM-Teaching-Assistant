from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=32000, show_progress = True )
tokenizer.pre_tokenizer = Whitespace()



file= ["sampled_OWT.txt"]


tokenizer.train(file,trainer)


tokenizer.save("tokenizer_2.json")
