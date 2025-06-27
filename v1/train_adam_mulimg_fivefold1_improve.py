import os, glob
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import torch
from torch.utils.data import DataLoader

# 1. 定义 collate_fn
def pad_collate_fn(batch):
    """
    batch: List of tuples [(imgs1, label1), (imgs2, label2), …]
      imgsᵢ: Tensor of shape (Nᵢ, 3, H, W)
      labelᵢ: intF
    返回:
      padded_imgs: FloatTensor (B, max_N, 3, H, W)
      labels:      LongTensor  (B,)
      mask:        FloatTensor (B, max_N)  1=真实图 0=padding
    """
    imgs_list, labels = zip(*batch)
    B = len(imgs_list)
    N_list = [imgs.shape[0] for imgs in imgs_list]
    max_N = max(N_list)

    C, H, W = imgs_list[0].shape[1:]
    padded = torch.zeros(B, max_N, C, H, W, dtype=imgs_list[0].dtype)
    mask   = torch.zeros(B, max_N, dtype=torch.float32)
    for i, imgs in enumerate(imgs_list):
        n = imgs.shape[0]
        padded[i, :n] = imgs
        mask[i, :n] = 1.0

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels, mask


# 1. 定义 Multi-Image Dataset
class MultiImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir/
          class0/
            sampleA/  <-- 这里是一个样本，里面多张图片
              *.jpg
            sampleB/
          class1/
            …
        """
        self.samples = []  # 每个元素: ( [img_path1, img_path2, …], label )
        classes = sorted(d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d)))
        self.class_to_idx = {cls:i for i,cls in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            for sample in os.listdir(cls_dir):
                samp_dir = os.path.join(cls_dir, sample)
                if os.path.isdir(samp_dir):
                    imgs = glob.glob(os.path.join(samp_dir, '*.jpg')) + \
                           glob.glob(os.path.join(samp_dir, '*.png'))
                    if not imgs:
                        continue
                    self.samples.append((imgs, self.class_to_idx[cls]))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]
        # 先读取、变换、再堆叠成一个 (N, C, H, W) 的张量
        imgs = []
        for p in img_paths:
            im = Image.open(p).convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            imgs.append(im)
        # imgs: List of (C,H,W) → Tensor (N,C,H,W)
        imgs = torch.stack(imgs, dim=0)
        return imgs, label


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# --- 1. 定义 AttentionPool ---
class AttentionPool(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, feats, mask):
        # feats: (B, N, D), mask: (B, N)
        h = self.act(self.fc1(feats))        # (B, N, hidden_dim)
        scores = self.fc2(h).squeeze(-1)     # (B, N)
        scores = scores.masked_fill(mask==0, -1e9)
        weights = torch.softmax(scores, dim=1)       # (B, N)
        pooled = (feats * weights.unsqueeze(-1)).sum(1)  # (B, D)
        return pooled, weights

# --- 2. 封装成一个 MultiImageModel ---
class MultiImageModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # a) 加载预训练 ResNet50 并移除 fc
        backbone = models.resnet50(pretrained=True)
        self.feat_dim = backbone.fc.in_features
        for p in backbone.parameters():
            p.requires_grad = False
        # 只解冻 layer3 和 layer4（可选）
        for name, p in backbone.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4"):
                p.requires_grad = True
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # 输出 (B*N, D,1,1)

        # b) 注意力池化
        self.pool = AttentionPool(self.feat_dim, hidden_dim=256)

        # c) 分类头
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, imgs, mask):
        """
        imgs: (B, N, C, H, W)
        mask: (B, N)
        """
        B, N, C, H, W = imgs.shape
        # 1) 拼成 (B*N, C, H, W)
        x = imgs.view(B*N, C, H, W)
        # 2) backbone 提取 (B*N, D,1,1)
        feats = self.backbone(x)
        feats = feats.view(B, N, self.feat_dim)  # (B, N, D)
        # 3) attention pooling
        pooled, attn_weights = self.pool(feats, mask)  # pooled: (B, D)
        # 4) 分类
        logits = self.classifier(pooled)               # (B, num_classes)
        return logits, attn_weights


# 2. 预处理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

data_dir = '/home/dalhxwlyjsuo/criait_tansy/project/resnet-50/data_sto/train_latest'
dataset = MultiImageFolder(data_dir, transform=None)
num_classes = len(dataset.class_to_idx)

# 3. 五折划分
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_acc = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n=== Fold {fold+1}/{k_folds} ===")

    # 构造两个子 Dataset，并分别设置 transform
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset   = torch.utils.data.Subset(dataset, val_idx)
    train_dataset.dataset.transform = train_tf
    val_dataset.dataset.transform   = val_tf

    # 2. 创建 DataLoader 时传入
    train_loader = DataLoader(
        train_dataset,
        #batch_size=4,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=pad_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        #batch_size=4,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=pad_collate_fn
    )

    # 4. 模型：ResNet50 backbone + 平均池化多图特征 + 分类头
    backbone = models.resnet50(pretrained=True)
    # 去掉最后的 fc
    feat_dim = backbone.fc.in_features
    backbone = nn.Sequential(*list(backbone.children())[:-1])  # 输出 (B*N, feat_dim,1,1)
    for p in backbone.parameters():
        p.requires_grad = False

    classifier = nn.Linear(feat_dim, num_classes)

    # model = nn.Module()
    # model.backbone = backbone
    # model.classifier = classifier
    # model = model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # #optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    # # 6) 优化器：只更新解冻的 backbone + 分类头
    # optim_params = [
    #     {"params": backbone[6:].parameters(), "lr": 1e-4},  # layer3/4 用小 lr
    #     {"params": classifier.parameters(),        "lr": 1e-3},  # 分类头 lr 大些
    # ]
    # optimizer = optim.Adam(optim_params)

    model = MultiImageModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.feat_dim and model.backbone[6:].parameters(), 'lr':1e-4},  # 如果部分解冻
        {'params': model.pool.parameters(), 'lr':1e-3},
        {'params': model.classifier.parameters(), 'lr':1e-3},
    ])


    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0

    # 5. 训练 & 验证
    for epoch in range(1, 31):
        model.train()
        for imgs_batch, labels, mask in train_loader:
            imgs_batch = imgs_batch.to(device)    # (B, N, 3, H, W)
            mask = mask.to(device)                # (B, N)
            logits, _ = model(imgs_batch, mask)
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证时同理，只要调用 forward 返回 logits 即可
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for imgs_batch, labels, mask in val_loader:
                imgs_batch = imgs_batch.to(device)
                mask = mask.to(device)
                logits, _ = model(imgs_batch, mask)
                pred = logits.argmax(dim=1)
                correct += (pred.cpu() == labels).sum().item()
                total += imgs_batch.size(0)
        acc = correct / total * 100
        
        print(f"Validation Accuracy: {acc:.2f}%")
        print(f"Epoch {epoch} val acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'/home/dalhxwlyjsuo/criait_tansy/project/resnet-50/checkpoint_mulimg/best_model_fold{fold+1}.pth')
            print("Best model saved for this fold.")

    fold_acc.append(best_acc)
    print(f"Best val acc for fold {fold+1}: {best_acc:.2f}%")

print("\n==== 5-Fold CV Results ====")
for i, a in enumerate(fold_acc, 1):
    print(f"Fold {i}: {a:.2f}%")
print(f"Average: {np.mean(fold_acc):.2f}%")
