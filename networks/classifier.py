import torch
import torch.nn as nn

from .image_encoder import EncoderResNet
from .language_encoder import EncoderBert

class MultiModalClassifier(nn.Module):
    def __init__(self, config):
        super(MultiModalClassifier, self).__init__()
        self.image_encoder = EncoderResNet(embed_dim=config.embed_dim, cnn_type=config.cnn_type)
        self.text_encoder = EncoderBert(config.embed_dim, txt_type=config.txt_type)
        
        # self.fc = nn.Linear(config.embed_dim, 14)
        self.fc = nn.Linear(2 * config.embed_dim, 14)

    
    def forward(self, tokenizer, img, txt):
        embed_img = self.image_encoder(img)
        embed_text = self.text_encoder(tokenizer, txt)
        concat_embed = torch.cat((embed_img["embedding"], embed_text["embedding"]), dim=1)
        out = self.fc(concat_embed)
        return {
            "logits": out,
            "image_features": embed_img['embedding'],
            "caption_features": embed_text['embedding'],
        }

class TextSplitClassifier(nn.Module):
    def __init__(self, config):
        super(TextSplitClassifier, self).__init__()
        self.text_encoder = EncoderBert(config.embed_dim, txt_type=config.txt_type)
        self.fc = nn.Linear(2 * config.embed_dim, 14)

    def forward(self, tokenizer, txt):
        embed_text = self.text_encoder(tokenizer, txt)
        embed_img = torch.zeros_like(embed_text["embedding"])

        concat_embed  = torch.cat((embed_img, embed_text["embedding"]), dim=1)
        out = self.fc(concat_embed)
        return {
            "logits":out,
            "image_features": embed_img,
            "caption_features":embed_text["embedding"]
        }

class ImageSplitClassifier(nn.Module):
    def __init__(self, config):
        super(ImageSplitClassifier, self).__init__()
        self.image_encoder = EncoderResNet(embed_dim=config.embed_dim, cnn_type=config.cnn_type)
        self.fc = nn.Linear(2 * config.embed_dim, 14)

    def forward(self, img):
        embed_img = self.image_encoder(img)
        embed_text = torch.zeros_like(embed_img["embedding"])
        concat_embed  = torch.cat((embed_img["embedding"], embed_text), dim=1)
        out = self.fc(concat_embed)
        return {
            "logits":out,
            "image_features": embed_img["embedding"],
            "caption_features":embed_text
        }

