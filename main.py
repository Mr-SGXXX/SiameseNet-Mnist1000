import argparse
import logging
import random
import torch
import numpy as np
import os
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from model import ResNet, Siamese, ContrastLoss
from dataset import MNIST_1000, AddGaussianNoise
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier

Train_transforms = transforms.Compose([
    transforms.ToTensor()
])
Test_transforms = transforms.Compose([
    AddGaussianNoise(0.0, 0.5),
    transforms.ToTensor()
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int, default=200, help="设置Batch Size")
    parser.add_argument('-dp', '--data_path', default='./data', help="数据集存放位置")
    parser.add_argument('-se', '--s_epoch', type=int, default=15, help="孪生网络迭代轮数")
    parser.add_argument('-re', '--r_epoch', type=int, default=0, help="ResNet迭代轮数")
    parser.add_argument('-lp', '--log_path', default="log/result.log", help="日志路径")
    parser.add_argument('-wp', '--weight_path', default="weight/weight.pth", help="权值路径")
    parser.add_argument('-d', '--device', default="cuda:0" if torch.cuda.is_available() else "cpu", help="使用的计算设备")
    parser.add_argument('-lr', '--learn_rate', type=float, default=1e-3, help="学习率")
    parser.add_argument('-op', '--optimizer', default="Adam", help="优化器设置")
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help="正则化项系数")
    return parser.parse_args()


def train_one_epoch_siamese(model, data_loader, optimizer, loss_func, device, logger):
    model.train()
    total_loss = 0.0
    for i, (x1, x2, label) in enumerate(data_loader):
        optimizer.zero_grad()
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)
        x1, x2, l1 = model(x1, x2)
        # loss = loss_func(x1, x2, label)
        loss = loss_func(l1, label.float())
        loss.backward()
        total_loss += loss.sum().detach().cpu().item()
        optimizer.step()
        if i % 100 == 99:
            logger.info(f"Siamese Iteration:{i + 1}/{len(data_loader)}\t Loss = {total_loss / (i + 1)}")
    return total_loss / len(data_loader)


def train_one_epoch_resnet(model, data_loader, optimizer, loss_func, device, logger):
    model.train()
    total_loss = 0.0
    for i, (x, label) in enumerate(data_loader):
        optimizer.zero_grad()
        x, label = x.to(device), label.to(device)
        _, x = model(x)
        loss = loss_func(x, label)
        loss.backward()
        total_loss += loss.sum().detach().cpu().item()
        optimizer.step()
        if i % 10 == 9:
            logger.info(f"Iteration:{i + 1}/{len(data_loader)}\t Loss = {total_loss / (i + 1)}")
    return total_loss / len(data_loader)


def model_train(model, logger, args):
    if os.path.exists(args.weight_path):
        model.load_state_dict(torch.load(args.weight_path))
        logger.info("Model Weight Load Over, Continue Train")
    else:
        logger.info("Train From Beginning")
    model.to(args.device)
    if args.r_epoch != 0:
        train_dataset = MNIST_1000(args.data_path, False, False, Train_transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    siamese_dataset = MNIST_1000(args.data_path, False, True, Train_transforms)
    siamese_dataloader = DataLoader(siamese_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    siamese = Siamese(model).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    # loss_func_a = ContrastLoss()
    loss_func_a = BCEWithLogitsLoss()
    loss_func_b = torch.nn.CrossEntropyLoss()
    for cur_epoch in range(args.s_epoch):
        loss = train_one_epoch_siamese(siamese, siamese_dataloader, optimizer, loss_func_a, args.device, logger)
        logger.info(f"Siamese Epoch: {cur_epoch + 1}/{args.s_epoch}\t Loss:{loss}")
        torch.save(model.state_dict(), args.weight_path)
    torch.save(model.state_dict(), "weight/Siamese_weight.pth")
    if args.s_epoch != 0:
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    for cur_epoch in range(args.r_epoch):
        loss = train_one_epoch_resnet(model, train_dataloader, optimizer, loss_func_b, args.device, logger)
        logger.info(f"Epoch: {cur_epoch + 1}/{args.r_epoch}\t Loss:{loss}")
        torch.save(model.state_dict(), args.weight_path)


def model_test(model, logger, args):
    model.load_state_dict(torch.load("./weight/Siamese_weight-15.pth"))
    model.eval()
    test_dataset = MNIST_1000(args.data_path, True, transform=Test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    loss_func = torch.nn.CrossEntropyLoss()
    loss_total = 0.0
    right_num = 0
    total_num = 0
    model.to(args.device)
    model.eval()
    train_dataset = MNIST_1000(args.data_path, False, False, Train_transforms)

    KNN = KNeighborsClassifier(45)
    with torch.no_grad():
        train_data = [model(data.view(1, *data.size()).to(args.device))[0].view(-1).cpu().detach().numpy() for data, _
                      in train_dataset]
        train_label = [label for _, label in train_dataset]
        test_data = [model(data.view(1, *data.size()).to(args.device))[0].view(-1).cpu().detach().numpy() for data, _ in
                     test_dataset]
        test_label = [label for _, label in test_dataset]
        KNN.fit(train_data, train_label)
        score = KNN.score(test_data, test_label)
        logger.info(f"Accuracy: {score}")
    if args.r_epoch != 0:
        model.load_state_dict(torch.load(args.weight_path))

        for img, label in test_dataloader:
            img = img.to(args.device)
            label = label.to(args.device)
            _, output = model(img)
            loss = loss_func(output, label)
            right_num += (torch.argmax(output, dim=1) == label).sum().item()
            total_num += output.size(0)
            loss_total += loss.item()
        logger.info(f"Loss: {loss_total / len(test_dataloader)}\tAccuracy: {right_num / total_num}")


def main():
    args = parse_args()
    if not os.path.exists(os.path.dirname(args.log_path)):
        os.mkdir(os.path.dirname(args.log_path))
    if not os.path.exists("./weight"):
        os.mkdir("weight")

    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s]-[%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # 让日志也输出在控制台上
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 固定随机种子
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = ResNet()
    # model_train(model, logger, args)
    model_test(model, logger, args)


if __name__ == "__main__":
    main()
