from __future__ import print_function

import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
import wandb

from torchvision import transforms
from util import adjust_learning_rate, AverageMeter, accuracy
from models.resnet import MyResNetsCMC
from models.LinearModel import LinearClassifierResNet
from dataset import TouchFolderLabel

# from spawn import spawn


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['resnet50t1', 'resnet101t1', 'resnet18t1',
                                                                         'resnet50t2', 'resnet101t2', 'resnet18t2',
                                                                         'resnet50t3', 'resnet101t3', 'resnet18t3'])
    parser.add_argument('--model_path', type=str, default=None, help='the model to test')
    parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')

    parser.add_argument('--test_modality', type=str, default='touch', choices=['touch', 'RGB'])

    # dataset
    parser.add_argument('--dataset', type=str, default='touch_and_go', choices=['touch_and_go', 'touch_hard', 'touch_rough'])

    # add new views
    parser.add_argument('--view', type=str, default='Touch', choices=['Touch'])

    # path definition
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--save_path', type=str, default=None, help='path to save linear classifier')

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    # log file
    parser.add_argument('--log', type=str, default='time_linear.txt', help='log file')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--no_ckpt', action='store_true', help='No ckpt')

    #wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb')
    parser.add_argument('--wandb_name', type=str, default=None, help='username of wandb')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.save_path is None):
        raise ValueError('one or more of the folders is None: data_folder | save_path')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.model_path.split('/')[-2]
    opt.model_name = 'calibrated_{}_bsz_{}_lr_{}_decay_{}'.format(opt.model_name, opt.batch_size, opt.learning_rate,
                                                                  opt.weight_decay)

    opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'touch_and_go':
        opt.n_label = 20
    if opt.dataset == 'touch_hard':
        opt.n_label = 2
    if opt.dataset == 'touch_rough':
        opt.n_label = 2

    return opt


def get_train_val_loader(args):
    data_folder = args.data_folder

    if args.view == 'Touch':
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]
    else:
        raise NotImplemented('view not implemented {}'.format(args.view))

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    if args.dataset == 'touch_and_go' or args.dataset == 'touch_rough' or args.dataset == 'touch_hard':
        if args.dataset == 'touch_hard':
            print('hard')
            train_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='train', label='hard')
            val_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='test', label='hard')
        elif args.dataset == 'touch_rough':
            print('rough')
            train_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='train', label='rough')
            val_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='test', label='rough')
        elif args.dataset == 'touch_and_go':
            train_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='train')
            val_dataset = TouchFolderLabel(data_folder, transform=train_transform, mode='test')
        else:
            raise NotImplementedError('dataset not supported {}'.format(args.dataset))

        print('number of train: {}'.format(len(train_dataset)))
        print('number of val: {}'.format(len(val_dataset)))

        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
    else:
        print('dataset unidentified')
        exit()

    return train_loader, val_loader, train_sampler


def set_model(args):
    if args.model.startswith('resnet'):
        model = MyResNetsCMC(args.model)
        if args.model.endswith('t2'):
            classifier = LinearClassifierResNet(args.layer, args.n_label)
        else:
            # Linear Probing for ResNet t1 and t3 has not been implemented
            raise NotImplementedError('model not supported {}'.format(args.model))
        print(classifier)
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    if args.no_ckpt == True:
         print('==> Train from scratch')
    
    # load pre-trained model
    else:
        print('==> loading pre-trained model')
        ckpt = torch.load(args.model_path)
        model.load_state_dict(ckpt['model'])
        print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
        print('==> done')

    model = model.cuda()
    classifier = classifier.cuda()

    model.eval()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    return model, classifier, criterion


def set_optimizer(args, classifier):
    optimizer = optim.SGD(classifier.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, classifier, criterion, optimizer, opt):
    """
    one epoch training
    """
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float()
        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
        target = target.cuda(opt.gpu, non_blocking=True)
        # ===================forward=====================
        with torch.no_grad():
            feat_image, feat_touch = model(input, opt.layer)
            if opt.test_modality == 'touch':
                feat = feat_touch.detach()
            else:
                feat = feat_image.detach()

        output = classifier(feat)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 1))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, classifier, criterion, opt):
    """
    evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    classifier.eval()
    # dis = torch.zeros(7)

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

            # compute output
            feat_image, feat_touch = model(input, opt.layer)
            if opt.test_modality == 'touch':
                feat = feat_touch
            else:
                feat = feat_image

            output = classifier(feat)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
    return top1.avg, top5.avg, losses.avg


def main():
    global best_acc1
    best_acc1 = 0

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # set the data loader
    train_loader, val_loader, train_sampler = get_train_val_loader(args)

    # set the model
    model, classifier, criterion = set_model(args)

    # set optimizer
    optimizer = set_optimizer(args, classifier)

    cudnn.benchmark = True

    # optionally resume linear classifier
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc1 = checkpoint['best_acc1']
            best_acc1 = best_acc1.cuda()
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # wandb
    if args.wandb:
        if args.dataset == 'touch_hard':
            print('log in touch hard')
            wandb.init(project="hard", entity=args.wandb_name, name=args.model_name)
        elif args.dataset == 'touch_rough':
            print('log in touch rough')
            wandb.init(project="rough", entity=args.wandb_name, name=args.model_name)
        else:
            print('log in linear cmc')
            wandb.init(project="material", entity=args.wandb_name, name=args.model_name)
        wandb.config = {
            "learning_rate": args.learning_rate,
            'epochs': args.epochs,
            "lr_decay_epochs": args.lr_decay_epochs,
            "batch_size": args.batch_size, 
            "lr_decay_rate": args.lr_decay_rate,
            }
        wandb.watch(classifier)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc5, train_loss = train(epoch, train_loader, model, classifier, criterion, optimizer, args)
        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        print("==> testing...")
        test_acc, test_acc5, test_loss = validate(val_loader, model, classifier, criterion, args)

        # wandb
        if args.wandb:
            wandb.log({"train_acc": train_acc, "train_loss": train_loss, "test_acc": test_acc, "test_loss": test_loss})

        # save the best model
        if test_acc > best_acc1:
            best_acc1 = test_acc
            state = {
                'opt': args,
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }
            save_name = '{}_layer{}.pth'.format(args.model, args.layer)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving best model!')
            torch.save(state, save_name)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'best_acc1': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_name = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving regular model!')
            torch.save(state, save_name)



if __name__ == '__main__':
    best_acc1 = 0
    main()
