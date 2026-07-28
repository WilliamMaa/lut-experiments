"""
Fashion-MNIST 知识蒸馏
Teacher: 轻量 CNN (~92%)
Student: 24-dim 全连接 (~88% base)
目标: 让 student 逼近 teacher 的软标签，突破小网络天花板
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# ==========================================
# Teacher: 轻量 CNN
# ==========================================
class TeacherCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ==========================================
# Student: 24-dim 全连接
# ==========================================
class StudentFC(nn.Module):
    def __init__(self, dim=24):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, dim),
            nn.ReLU(),
            nn.Linear(dim, 20),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def train_teacher(model, trainloader, testloader, device, epochs=30):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for imgs, labels in tqdm(trainloader, desc=f"Teacher Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 5 == 0:
            print(f"  [Teacher Epoch {epoch+1:2d}] Test: {acc:.2f}%")

    print(f"Teacher 最佳准确率: {best_acc:.2f}%")
    return best_acc


def train_student_distill(student, teacher, trainloader, testloader, device, epochs=100, alpha=0.7, T=4):
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_loss = nn.CrossEntropyLoss()
    best_acc = 0.0

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for imgs, labels in tqdm(trainloader, desc=f"Student Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            student_logits = student(imgs)
            with torch.no_grad():
                teacher_logits = teacher(imgs)

            # Hard label loss
            loss_hard = ce_loss(student_logits, labels)
            # Soft label loss (KL divergence)
            loss_soft = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * (T * T)

            loss = alpha * loss_hard + (1 - alpha) * loss_soft
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = student(imgs)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0:
            print(f"  [Student Epoch {epoch+1:3d}] Test: {acc:.2f}%")

    return best_acc


def train_student_baseline(student, trainloader, testloader, device, epochs=100):
    """直接训练 student，作为对比"""
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        student.train()
        for imgs, labels in tqdm(trainloader, desc=f"Baseline Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = student(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = student(imgs)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0:
            print(f"  [Baseline Epoch {epoch+1:3d}] Test: {acc:.2f}%")

    return best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    trainloader = DataLoader(
        torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=256, shuffle=True
    )
    testloader = DataLoader(
        torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform),
        batch_size=256, shuffle=False
    )

    # ===== 阶段 1：训练或加载 Teacher =====
    teacher = TeacherCNN().to(device)
    try:
        teacher.load_state_dict(torch.load('fashion_teacher_cnn.pth', map_location=device, weights_only=False))
        print("加载已有 Teacher 权重")
        teacher.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = teacher(imgs)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        teacher_acc = 100.0 * correct / total
        print(f"Teacher 测试集: {teacher_acc:.2f}%")
    except FileNotFoundError:
        print("\n>>> 训练 Teacher CNN...")
        teacher_acc = train_teacher(teacher, trainloader, testloader, device, epochs=30)
        torch.save(teacher.state_dict(), 'fashion_teacher_cnn.pth')
        print(f"Teacher 已保存，参数量: {sum(p.numel() for p in teacher.parameters()):,}")

    # ===== 阶段 2：Student 直接训练（对比基线）=====
    print("\n>>> Student 直接训练（无蒸馏）...")
    student_baseline = StudentFC(dim=24).to(device)
    baseline_acc = train_student_baseline(student_baseline, trainloader, testloader, device, epochs=100)

    # ===== 阶段 3：Student 蒸馏训练 =====
    print("\n>>> Student 知识蒸馏...")
    student_distill = StudentFC(dim=24).to(device)
    distill_acc = train_student_distill(student_distill, teacher, trainloader, testloader, device, epochs=100, alpha=0.7, T=4)

    print("\n" + "=" * 60)
    print("【知识蒸馏结果】")
    print("=" * 60)
    print(f"Teacher CNN 准确率:     {teacher_acc:.2f}%")
    print(f"Student 直接训练:       {baseline_acc:.2f}%")
    print(f"Student 知识蒸馏:       {distill_acc:.2f}%")
    print(f"蒸馏增益:               {distill_acc - baseline_acc:+.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
