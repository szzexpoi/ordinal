import sys
sys.path.append('./util')
sys.path.append('./model')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import LSP_generator
from ordinal_net import OrdinalNet_slim, OrdinalNet_att, OrdinalNet_feat
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import numpy as np
import cv2
import argparse
import os
import time
import gc
import tensorflow as tf
from loss import bce_loss, cross_entropy
import json

parser = argparse.ArgumentParser(description='Ordinal prediction on the LSP dataset')
parser.add_argument('--mode', type=str, default='train', help='Selecting running mode (default: train)')
parser.add_argument('--anno_dir', type=str, default='./annotation', help='Directory to annotation files')
parser.add_argument('--img_dir', type=str, default='./image', help='Directory to image files')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory for saving checkpoint')
parser.add_argument('--weights', type=str, default=None, help='Trained model to be loaded (default: None)')
parser.add_argument('--epoch', type=int, default=120, help='Defining maximal number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Defining initial learning rate (default: 1e-3)')
parser.add_argument('--batch_size', type=int, default=32, help='Defining batch size for training (default: 32)')
parser.add_argument('--clip', type=float, default=0, help='Gradient clipping to prevent gradient explode (default: 0)')
parser.add_argument('--bbox_size', type=int, default=16, help='Defining size of the bounding box')
parser.add_argument('--patch_size', type=int, default=16, help='Defining size of patch for each point')
parser.add_argument('--mask_size', type=int, default=32, help='Defining size of gaussian mask for each point')
parser.add_argument('--mask_sigma', type=float, default=0.2, help='Standard deviation for creating the gaussian mask')
parser.add_argument('--use_softmax', type=bool, default=False, help='using softmax activation for output or not')

args = parser.parse_args()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(init_lr,optimizer, epoch):
    "adatively adjust lr based on epoch"
    lr = init_lr * (0.25 ** int((epoch+1)/50)) #previously 0.25/40

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_dir)

    # define dataloader
    train_data = LSP_generator(args.img_dir, args.anno_dir, 'train', args.bbox_size,
                            args.patch_size, args.mask_size, args.mask_sigma, args.use_softmax)
    val_data = LSP_generator(args.img_dir, args.anno_dir, 'valid', args.bbox_size,
                            args.patch_size, args.mask_size, args.mask_sigma, args.use_softmax)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                    shuffle=True, num_workers=12)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)

    # initialize model
    # model = OrdinalNet_slim() # start with a simple one
    model = OrdinalNet_att(args.use_softmax)
    # model = OrdinalNet_feat(args.use_softmax)
    model = model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) #1e-8

    def train(iteration):
        """ Training process for a single epoch.
        """
        model.train()
        avg_loss = 0

        for batch_idx,(BB, P1, P2, M1, M2, pos, label) in enumerate(trainloader):
            BB, P1, P2, M1, M2, pos, label = BB.cuda(), P1.cuda(), P2.cuda(), M1.cuda(), M2.cuda(), pos.cuda(), label.cuda()
            optimizer.zero_grad()

            # prediction = model(BB, P1, P2, M1, M2)
            prediction = model(BB, M1, M2, pos)

            if not args.use_softmax:
                loss = bce_loss(prediction, label)
            else:
                loss = cross_entropy(prediction, label)
            loss.backward()

            if not args.clip == 0 :
                clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()
            avg_loss = (avg_loss*np.maximum(0, batch_idx) +
                        loss.data.cpu().numpy())/(batch_idx+1)

            if batch_idx%25 == 0:
                with tf_summary_writer.as_default():
                    tf.summary.scalar('average training loss',avg_loss,step=iteration)

            iteration += 1

        return iteration

    def test(iteration):
        """ Function for validation.
        """
        model.eval()
        acc = []

        for batch_idx,(BB, P1, P2, M1, M2, pos, label) in enumerate(valloader):
            BB, P1, P2, M1, M2, pos = BB.cuda(), P1.cuda(), P2.cuda(), M1.cuda(), M2.cuda(), pos.cuda()
            # prediction = model(BB, P1, P2, M1, M2)
            prediction = model(BB, M1, M2, pos)

            if not args.use_softmax:
                prediction  = prediction.squeeze(-1).data.cpu().numpy()
                label = label.squeeze(-1).data.numpy()
                prediction[prediction>=0.5] = 1
                prediction[prediction<0.5] = 0
            else:
                prediction = prediction.argmax(-1).data.cpu().numpy()
                label = label.argmax(-1).data.cpu().numpy()
            acc.extend(prediction==label)

        acc = np.mean(np.array(acc))

        with tf_summary_writer.as_default():
            tf.summary.scalar('Accuracy', acc, step=iteration)
        return acc


    #main loop for training:
    print('Start training model')
    iteration = 0
    val_acc = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr,optimizer, epoch)
        iteration = train(iteration)
        cur_score = test(iteration)

        #save the best check point and latest checkpoint
        if cur_score > val_acc:
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model_best.pth'))
            val_acc = cur_score
        torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model.pth'))


def eval():
    # define dataloader
    test_data = LSP_generator(args.img_dir, args.anno_dir, 'test', args.bbox_size,
                            args.patch_size, args.mask_size, args.mask_sigma, args.use_softmax)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)

    # initialize model
    # model = OrdinalNet_slim() # start with a simple one
    model = OrdinalNet_att(args.use_softmax)
    model.load_state_dict(torch.load(args.weights))
    model = model.cuda()
    model.eval()

    acc = []

    for batch_idx,(BB, P1, P2, M1, M2, pos, label) in enumerate(testloader):
        BB, P1, P2, M1, M2, pos = BB.cuda(), P1.cuda(), P2.cuda(), M1.cuda(), M2.cuda(), pos.cuda()
        # prediction = model(BB, P1, P2, M1, M2)
        prediction = model(BB, M1, M2, pos)
        # print(prediction)

        if not args.use_softmax:
            prediction  = prediction.squeeze(-1).data.cpu().numpy()
            label = label.squeeze(-1).data.numpy()
            prediction[prediction>=0.5] = 1
            prediction[prediction<0.5] = 0
        else:
            prediction = prediction.argmax(-1).data.cpu().numpy()
            label = label.argmax(-1).data.cpu().numpy()
        acc.extend(prediction==label)

    acc = np.mean(np.array(acc))
    print('Test accuracy is %.3f' %acc)

if args.mode == 'train':
    main()
else:
    eval()
