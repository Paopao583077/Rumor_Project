import torch
import torch.nn as nn
from transformers import BertModel
import torchvision.models as models

# ==========================================
# 模型 1：单模态文本模型 (Text-Only BERT)
# ==========================================
class TextOnlyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(TextOnlyModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        
        # --- [同步冻结逻辑] ---
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        # --------------------

        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, image=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = bert_out.pooler_output
        logits = self.classifier(text_feat)
        return logits

# ==========================================
# 模型 2：单模态图像模型 (Image-Only ResNet)
# ==========================================
class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageOnlyModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet_features = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids=None, attention_mask=None, image=None):
        img_feat_map = self.resnet_features(image)
        img_feat = self.avgpool(img_feat_map).view(image.size(0), -1)
        logits = self.classifier(img_feat)
        return logits

# ==========================================
# 模型 3：简单拼接模型 (BERT + ResNet Concat)
# ==========================================
class ConcatModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ConcatModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        
        # --- [同步冻结逻辑] ---
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        # --------------------

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet_features = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = bert_out.pooler_output
        
        img_feat_map = self.resnet_features(image)
        img_feat = self.avgpool(img_feat_map).view(image.size(0), -1)
        
        concat_feat = torch.cat((text_feat, img_feat), dim=1)
        logits = self.classifier(concat_feat)
        return logits
