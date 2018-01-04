import os
import logging
import argparse
import torch
import torch.backends.cudnn as cudnn
from model import densenet
from utils.visualize import Dashboard
from utils.misc import save_checkpoint, adjust_learning_rate
from utils.loss import RewardLoss, ClassifierLoss
from Trainer import train, validate
from torchvision import datasets, transforms
torch.manual_seed(10)

parser = argparse.ArgumentParser(description='Code framework')
parser.add_argument('--env', type=str,help='visdom environment')
parser.add_argument('--data-dir', default='data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='resume from the latest checkpoint')
parser.add_argument('--resume-path', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--draw-freq', default=1, type=int, metavar='N',
                    help='draw results every draw_freq samples')
parser.add_argument('--print-freq', default=1, type=int, metavar='N',
                    help='draw results every draw_freq samples')

best_loss1 = 1e6
cudnn.benchmark = True


def main():
    logger = logging.getLogger('main')
    global args, best_loss1
    args = parser.parse_args()
    vplt = Dashboard(server='http://127.0.0.1', port=8099, env=args.env)

    if not os.path.exists(os.path.join('checkpoints', args.env)):
        os.makedirs(os.path.join('checkpoints', args.env))

    # create model
    model = densenet()

    # loss functions
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if args.resume_path is None:
            args.resume_path = os.path.join('checkpoints', args.env, 'latest.pth.tar')

        if os.path.isfile(args.resume_path):
            logger.info("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            args.start_epoch = checkpoint['epoch']
            best_loss1 = checkpoint['best_loss1']
            model.load_state_dict(checkpoint['state_dict'])
            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
            train_acc = checkpoint['train_acc']
            val_acc = checkpoint['train_acc']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume_path))
    else:
        train_loss = {}
        val_loss = {}
        train_acc = {}
        val_acc = {}


    logger.info('=> loading dataset')

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('dataset/data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           ])),
                       batch_size=args.batch_size, shuffle=True, num_workers=5)

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('dataset/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=5)

    if args.evaluate:
        validate(args=args, val_loader=val_loader, criterion1=criterion, model=model)
        return

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, args.lr, epoch)

        # train for one epoch
        train_loss_epoch, train_acc_epoch = train(args=args, train_loader=train_loader,
                                                model=model, optimizer=optimizer,
                                                criterion=criterion,
                                                epoch=epoch)

        # evaluate on validation set
        val_loss_epoch, val_acc_epoch, = validate(args=args, val_loader=val_loader,
                                               criterion1=criterion,
                                               model=model)

        train_loss[epoch] = train_loss_epoch
        val_loss[epoch] = val_loss_epoch
        train_acc[epoch] = train_acc_epoch
        val_acc[epoch] = val_acc_epoch

        # visualization
        vplt.draw(train_loss, val_loss, 'Loss')
        vplt.draw(train_acc, val_acc, 'Accuracy')

        # remember best loss and save checkpoint
        is_best = val_loss_epoch < best_loss1
        best_loss1 = min(val_loss_epoch, best_loss1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss1': best_loss1,
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, is_best,
           filename='epoch_{}.pth.tar'.format(epoch+1),
           dir=os.path.join('checkpoints', args.env), epoch=epoch)


if __name__ == '__main__':
    main()