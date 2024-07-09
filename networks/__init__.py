from transformers import  BertTokenizer
from .classifier import MultiModalClassifier, TextSplitClassifier, ImageSplitClassifier

def get_mmclf(config):
    return MultiModalClassifier(config=config)

def get_clf(config, modality):
    if modality == 'text':
        return TextSplitClassifier(config)
    elif modality == 'image':
        return ImageSplitClassifier(config)


def get_tokenizer(config):
    if config.txt_type == 'bert-base-uncased':
        return BertTokenizer.from_pretrained("bert-base-uncased")
    if config.txt_type == 'tiny-bert':
        return BertTokenizer.from_pretrained("huawei-noah/TinyBERT_4L_zh")
