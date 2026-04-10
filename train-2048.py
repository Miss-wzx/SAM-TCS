from torch import optim
from dataset2048 import YSDataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from module import *
from history import History
import torch.nn as nn
import numpy as np

import time


def parse_args():
    # PARAMETERS
    parser = argparse.ArgumentParser('aec')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    return parser.parse_args()


def main(args):
    # 记录程序开始时间
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.is_available())
    # 读取数据
    data_path = './data2048.pklh5'

    train_dataset = YSDataset(data_path, t='train')
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    # 实例化模型
    m = CSTransformer1D(2048, 256).to(device)

    info = ''
    h = History('TCS', './history_save', args.learning_rate, args.epoch, args.batch_size, info=info)

    # 优化器选择和动态学习率
    optimizer = optim.Adam(m.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    # 训练模型
    for epoch in range(args.epoch):
        print('[%d/%d: ' % (epoch + 1, args.epoch) + '-' * 30 + ' ]')
        train_loss_sum = 0

        for batch_id, (x) in enumerate(train_data_loader, 0):
            optimizer.zero_grad()
            m = m.train()
            x = x.to(device)
            x_hat, _ = m(x)

            loss = F.mse_loss(x, x_hat)
            train_loss_sum += loss.item()

            loss.backward()
            optimizer.step()

            print('[%d/%d: %d/%d] loss: %f' % (epoch + 1, args.epoch, batch_id + 1, len(train_data_loader), loss.item()))

        scheduler.step()
        h.train_loss.append(train_loss_sum / len(train_data_loader))
        torch.save(m.cpu().state_dict(), './model/TCS.pth'.format(info))
        h.save_history()
        m = m.to(device)

    # 保存网络模型
    torch.save(m.cpu().state_dict(), './model/TCS.pth'.format(info))

    # 记录程序结束时间
    end_time = time.time()

    # 计算运行时间，单位为秒
    execution_time = end_time - start_time
    print(f"程序运行时间：{execution_time} 秒")

    h.execution_time = execution_time
    h.save_history()


if __name__ == '__main__':
    a = parse_args()
    main(a)
