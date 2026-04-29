import torch
import torch.nn as nn
from transformers import BertModel
import torchvision.models as models


# --- 核心模块：文本引导的视觉注意力 ---
class TextGuidedAttention(nn.Module):
    def __init__(self, text_dim, img_dim, hidden_dim):
        super(TextGuidedAttention, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.v_proj = nn.Linear(img_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text_feat, img_feat):
        t_proj = self.text_proj(text_feat).unsqueeze(1)
        i_proj = self.img_proj(img_feat)
        combined = self.tanh(t_proj + i_proj)
        scores = self.attention_score(combined).squeeze(-1)
        alpha = self.softmax(scores)

        v = self.v_proj(img_feat)
        context_img = (alpha.unsqueeze(-1) * v).sum(dim=1)
        return context_img


# --- 完整模型 ---
class RumorDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(RumorDetector, self).__init__()
        # 1. 文本专家 BERT
        self.bert = BertModel.from_pretrained('bert-base-chinese')

        # --- [新增] 冻结 BERT 底层参数逻辑 ---
        # 冻结所有参数
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 仅解冻最后一层 (Layer 11) 和 Pooler 层
        # 这样既保留了 BERT 的基础能力，又能针对任务进行微调
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        # ------------------------------------

        # 2. 视觉专家 ResNet
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet_features = nn.Sequential(*modules)

        # 3. 融合模块 (回退至 256 维)
        self.fusion_attention = TextGuidedAttention(768, 2048, 256)

        # 4. 最终裁判 (分类器)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = bert_out.pooler_output

        img_feat_map = self.resnet_features(image)
        batch, c, h, w = img_feat_map.size()
        img_feat_seq = img_feat_map.view(batch, c, -1).permute(0, 2, 1)

        attended_img = self.fusion_attention(text_feat, img_feat_seq)

        final_feat = torch.cat((text_feat, attended_img), dim=1)
        logits = self.classifier(final_feat)

        return logits
