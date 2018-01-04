import logging
import torch
import os
import sys
import time
from utils.misc import AverageMeter
from torch.autograd import Variable


def train(args, train_loader, model, optimizer, criterion, epoch):
    logger = logging.getLogger('train')

    log_dir = os.path.join('log', args.env)
    if not os.path.isdir(log_dir):
        logger.info('log dir does not exist, create log dir.')
        os.makedirs(log_dir)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='a+')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accu = AverageMeter()

    model = model.train()
    end = time.time()
    optimizer = optimizer

    for i, (image, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        image = image.cuda(async=True)
        target = target.cuda(async=True)
        image_var, target_var = Variable(image), Variable(target)

        # compute output
        output = model(image_var)

        loss = criterion(output, target_var)

        # update the cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = 10
        losses.update(loss.data[0])
        accu.update(acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % args.print_freq) == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'accuracy {accu.val:.4f} ({accu.avg:.4f})\t'
                        .format(
                         epoch + 1, i + 1, len(train_loader),
                         batch_time=batch_time, data_time=data_time, loss=losses, accu=accu))
            sys.stdout.flush()
            print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'accuracy {accu.val:.4f} ({accu.avg:.4f})\t'
                        .format(
                         epoch + 1, i + 1, len(train_loader),
                         batch_time=batch_time, data_time=data_time, loss=losses, accu=accu))

    logger.info(' * Loss: {losses.avg:.3f} accuracy:{accu.avg:.3f}'
                .format(losses=losses, accu=accu))
    print(' * Loss: {losses.avg:.3f} accuracy:{accu.avg:.3f}'.format(losses=losses, accu=accu))
    return losses.avg, accu.avg


def validate(args, val_loader, model, criterion, criterion2):

    logger = logging.getLogger('val')
    log_dir = os.path.join('log', args.env)
    if not os.path.isdir(log_dir):
        logger.info('log dir does not exist, create log dir.')
        os.makedirs(log_dir)
    fh = logging.FileHandler(os.path.join(log_dir, 'val.log'), mode='a+')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    batch_time = AverageMeter()
    losses = AverageMeter()
    accu = AverageMeter()

    # switch to evaluate mode
    model = model.eval()

    end = time.time()
    for i, (image, target) in enumerate(val_loader):

        # measure data loading time
        image = image.cuda(async=True)
        target = target.cuda(async=True)
        image_var, target_var =Variable(image, volatile=True), Variable(target, volatile=True)


        # compute output
        output = model(image_var)
        loss = criterion(output, target_var)
        acc = 10
        losses.update(loss.data[0])
        accu.update(acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % args.print_freq) == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'accuracy {accu.val:.4f} ({accu.avg:.4f})\t'
                        .format(i + 1, len(val_loader), batch_time=batch_time, loss=losses, accu=accu))
            sys.stdout.flush()
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'accuracy {accu.val:.4f} ({accu.avg:.4f})\t'
                        .format(i + 1, len(val_loader), batch_time=batch_time, loss=losses, accu=accu))

    logger.info(' * Loss: {losses.avg:.3f} accuracy:{accu.avg:.3f}'
                .format(losses=losses, accu=accu))
    print(' * Loss: {losses.avg:.3f} accuracy:{accu.avg:.3f}'
                .format(losses=losses, accu=accu))
    return losses.avg, accu.avg

