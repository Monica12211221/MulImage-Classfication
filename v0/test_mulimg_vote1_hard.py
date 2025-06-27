import os
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class_names = ['C1', 'C2', 'C3', 'O1', 'O2', 'O3', '非萎缩']
num_classes = len(class_names)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.resnet50(pretrained=False)
in_feats = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(in_feats, num_classes)
)
checkpoint_path = "/home/dalhxwlyjsuo/criait_tansy/project/resnet-50/checkpoint2_unfreeze/best_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

parent_dir = "/home/dalhxwlyjsuo/criait_tansy/project/resnet-50/data_sto/test/非萎缩-merge"
group_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
              if os.path.isdir(os.path.join(parent_dir, d))]
img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ----------- 定义辅助函数：屏幕+文件双写 -----------
def log(line, f):
    print(line)
    f.write(line + '\n')

save_txt = r"/home/dalhxwlyjsuo/criait_tansy/project/resnet-50/results_mul/group_vote_result_非萎缩_hard.txt"  # 输出文件路径（可以自定义）
with open(save_txt, 'w', encoding='utf-8') as fout:
    for group_dir in group_dirs:
        all_image_paths = []
        for fname in os.listdir(group_dir):
            ext = os.path.splitext(fname)[-1].lower()
            if ext in img_extensions:
                all_image_paths.append(os.path.join(group_dir, fname))

        if len(all_image_paths) == 0:
            log(f"在目录 {group_dir} 下未找到图片，跳过。", fout)
            continue

        all_predictions = []
        all_confidences = []

        with torch.no_grad():
            for img_path in all_image_paths:
                img = Image.open(img_path).convert("RGB")
                x = data_transforms(img).unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                prob_np = probs.cpu().numpy().squeeze()

                pred_idx = int(np.argmax(prob_np))
                pred_conf = float(prob_np[pred_idx])

                all_predictions.append(pred_idx)
                all_confidences.append(pred_conf)


        all_confidences = np.array(all_confidences)
        mean_confidence = float(np.mean(all_confidences))

        # ==== 硬投票 ====
        group_scores = np.zeros(num_classes, dtype=np.float32)
        for pred_cls in all_predictions:
            group_scores[pred_cls] += 1  # 每个图片的“票”权重相同，+1

        group_pred_idx = int(np.argmax(group_scores))
        group_pred_name = class_names[group_pred_idx]


        log(f"\n=== 组别：{os.path.basename(group_dir)} ===", fout)
        log(f"各类别累积分数：", fout)
        for i, cname in enumerate(class_names):
            log(f"  {cname:<10} | {group_scores[i]:.4f}", fout)
        log(f">>> 该组整体预测类别：{group_pred_name} (Index={group_pred_idx})", fout)
        log(f"图片总数：{len(all_image_paths)}，平均置信度：{mean_confidence:.4f}", fout)
        log("======================================\n", fout)
