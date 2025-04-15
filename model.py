import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 数据预处理模块
class MultimodalDataset(Dataset):
    def __init__(self, seq_length=300):
        self.seq_length = seq_length
        
    def __len__(self):
        return 1000  # 示例数据集大小
    
    def __getitem__(self, idx):
        # 模拟多模态数据生成
        # EEG: (C, T) -> (64, 300)
        eeg = torch.randn(64, self.seq_length)  
        # EMG: (M, T) -> (8, 300)
        emg = torch.abs(torch.randn(8, self.seq_length))  
        # EOG: (H, W) + stats -> (256, 256) + (6,)
        eog_img = torch.rand(256, 256)  
        eog_stats = torch.rand(6)
        
        # 标签
        workload = torch.rand(1)  # 回归目标
        class_label = torch.randint(0, 3, (1,))  # 分类目标
        
        return {
            'eeg': eeg,
            'emg': emg,
            'eog_img': eog_img,
            'eog_stats': eog_stats,
            'workload': workload,
            'class_label': class_label
        }

# EEG处理分支
class EEGBranch(nn.Module):
    def __init__(self, electrode_layout=(8, 8)):
        super().__init__()
        self.layout = electrode_layout
        
        # 3D卷积参数调整
        self.conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels=1,          # 输入通道
                out_channels=32,        # 输出通道
                kernel_size=(3, 3, 5),  # 调整核尺寸 (depth, height, width)
                padding=(1, 1, 2)       # 对应padding
            ),
            nn.GELU(),
            nn.MaxPool3d((1, 2, 2)),    # 调整池化区域
            nn.BatchNorm3d(32)
        )
        
        self.lstm = nn.LSTM(
            input_size=32 * (self.layout[0]//2) * (self.layout[1]//2),
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        # 输入x形状: (B, 64, 300)
        B, C, T = x.size()
        
        # 转换为3D拓扑图 (示例使用8x8布局)
        x = x.view(B, 1, self.layout[0], self.layout[1], T)  # (B, 1, 8, 8, 300)
        
        # 3D卷积处理
        x = self.conv3d(x)  # 输出形状 (B, 32, 8, 4, 300)
        
        # 展平空间维度
        x = x.permute(0, 4, 1, 2, 3)  # (B, 300, 32, 4, 4)
        x = x.reshape(B, T, -1)        # (B, 300, 32*4*4)
        
        # LSTM处理时序
        x, _ = self.lstm(x)
        return x  # 输出形状 (B, 300, 128)

# EMG处理分支
class EMGBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(8, 16, 5, padding=2),
            nn.PReLU(),
            nn.Conv1d(16, 16, 3, padding=1, groups=16),
            nn.Conv1d(16, 32, 1),
            nn.BatchNorm1d(32)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, x):
        x = self.conv(x)        # (B,32,300)
        x = x.permute(0, 2, 1)  # (B,300,32)
        x = self.transformer(x)
        return x  # 保持 (B,300,32)

# EOG处理分支 
class EOGBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # 图像分支
        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128
            nn.Conv2d(32, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d((16, 16))  # 修正为元组
        )
        # 统计分支
        self.stats_fc = nn.Linear(6, 16)
        # 融合
        self.fusion = nn.Linear(64*16*16 + 16, 128)
        
    def forward(self, img, stats):
        # img: (B, 256, 256) -> (B, 1, 256, 256)
        img_feat = self.img_encoder(img.unsqueeze(1))
        img_feat = img_feat.flatten(1)  # (B, 64*16*16)
        stats_feat = F.relu(self.stats_fc(stats))
        fused = torch.cat([img_feat, stats_feat], dim=1)
        return self.fusion(fused)  # (B, 128)

# 跨模态注意力融合模块
class CrossModalAttention(nn.Module):
    def __init__(self, input_dim=128, num_heads=4):
        super().__init__()
        # 统一使用相同输入维度
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True  # 重要参数
        )

    def forward(self, eeg_feat, emg_feat, eog_feat):
        # 输入形状均为 (B, 300, 128)
        q = self.query(eeg_feat)
        k = self.key(emg_feat)
        v = self.value(eog_feat)
        
        # 注意维度顺序 (B, Seq, Dim)
        att_out, _ = self.mha(q, k, v)
        return att_out

# 完整模型
class WorkloadModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 各模态分支
        self.eeg_branch = EEGBranch()  # 输出 (B,300,128)
        self.emg_branch = EMGBranch()  # 输出 (B,300,32)
        self.eog_branch = EOGBranch()  # 输出 (B,128)
        
        # 新增投影层
        self.emg_proj = nn.Linear(32, 128)
        self.eog_expand = nn.Linear(128, 128)
        
        # 注意力模块
        self.cross_att = CrossModalAttention(input_dim=128)
        
        # 回归头
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid())
        
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1))
        
    def forward(self, inputs):
        # 获取各模态特征
        eeg_feat = self.eeg_branch(inputs['eeg'])  # (B,300,128)
        emg_feat = self.emg_branch(inputs['emg'])  # (B,300,32)
        eog_feat = self.eog_branch(inputs['eog_img'], inputs['eog_stats'])  # (B,128)
        
        # 维度对齐
        emg_feat = self.emg_proj(emg_feat)  # (B,300,32)→(B,300,128)
        eog_feat = eog_feat.unsqueeze(1).expand(-1, 300, -1)  # (B,128)→(B,300,128)
        eog_feat = self.eog_expand(eog_feat)  # 确保特征维度
        
        # 注意力融合
        fused = self.cross_att(eeg_feat, emg_feat, eog_feat)
        fused_last = fused[:, -1, :]
        
        # 多任务输出
        workload = self.reg_head(fused_last)
        cls_probs = self.cls_head(fused_last)
        
        return {'workload': workload, 'class': cls_probs}

# 自定义损失函数
class MultitaskLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.reg_loss = nn.HuberLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        reg_loss = self.reg_loss(outputs['workload'], targets['workload'])
        cls_loss = self.cls_loss(outputs['class'], targets['class_label'].squeeze())
        return self.alpha * reg_loss + (1 - self.alpha) * cls_loss

# 训练流程示例
def train_model():
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WorkloadModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = MultitaskLoss()
    
    # 数据加载
    dataset = MultimodalDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 检查模型参数
    print("Model Structure:")
    print(model)
    
    # 验证前向传播
    test_input = {
        'eeg': torch.randn(2, 64, 300).to(device),
        'emg': torch.randn(2, 8, 300).to(device),
        'eog_img': torch.rand(2, 256, 256).to(device),
        'eog_stats': torch.rand(2, 6).to(device)
    }
    output = model(test_input)
    print("\nTest Output Shapes:")
    print(f"Workload: {output['workload'].shape}")
    print(f"Class: {output['class'].shape}")
    input("Press Enter to continue...")

    
    # 训练循环
    for epoch in range(500):
        model.train()
        total_loss = 0
        
        for batch in loader:
            # 数据迁移到设备
            inputs = {k: v.to(device) for k, v in batch.items() 
                     if k not in ['workload', 'class_label']}
            targets = {
                'workload': batch['workload'].to(device),
                'class_label': batch['class_label'].to(device)
            }
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        
        
def validate_model():
    model = WorkloadModel()
    test_input = {
        'eeg': torch.randn(2, 64, 300),
        'emg': torch.randn(2, 8, 300),
        'eog_img': torch.rand(2, 256, 256),
        'eog_stats': torch.rand(2, 6)
    }
    output = model(test_input)
    print(f"Workload输出形状: {output['workload'].shape}")  # 应为(2,1)
    print(f"Class输出形状: {output['class'].shape}")        # 应为(2,3)

def validate_dimensions():
    model = WorkloadModel()
    
    # 打印模型结构
    print("EMG投影层参数:", model.emg_proj)
    print(f"权重矩阵形状: {model.emg_proj.weight.shape}")  # 应为torch.Size([128, 32])
    
    # 测试前向传播
    test_input = {
        'eeg': torch.randn(2, 64, 300),
        'emg': torch.randn(2, 8, 300),
        'eog_img': torch.rand(2, 256, 256),
        'eog_stats': torch.rand(2, 6)
    }
    
    # 检查中间输出
    eeg_out = model.eeg_branch(test_input['eeg'])
    print("\nEEG分支输出形状:", eeg_out.shape)  # (2,300,128)
    
    emg_out = model.emg_branch(test_input['emg'])
    print("EMG分支输出形状:", emg_out.shape)   # (2,300,32)
    
    emg_proj = model.emg_proj(emg_out)
    print("EMG投影后形状:", emg_proj.shape)  # (2,300,128)
    
if __name__ == "__main__":
    validate_dimensions()
    validate_model()
    train_model()