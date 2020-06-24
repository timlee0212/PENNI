###################################################################
#   Provide essential utilities for model training, evaluation and analysis
#   Part of the filter decomposing project
#
#   UNFINISHED RESEARCH CODE
#   DO NOT DISTRIBUTE
#
#   For ICML 2020 Submission
#   Author: XXXXXXXXXXXXXXXXXXX
#   Date:   XXXXXXXXXXXXXXXXXXX
#
#   Changelog:
#   2020-01-22 Merge the decomposing utilities.
##################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets

from decompose.decomConv import DecomposedConv2D
from models.resnet_s import LambdaLayer
from tensorboardX import SummaryWriter

import numpy as np
import time
import os
import copy

def train_cifar10(model, epochs=100, batch_size=128, lr=0.01, reg=5e-4,
                  checkpoint_path = '', spar_reg = None, spar_param = 0.0,
                  scheduler='step', finetune=False, cross=False, cross_interval=5):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=True)
    if scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    elif scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    _train(model, trainloader, testloader, optimizer, epochs,
           scheduler, checkpoint_path, finetune=finetune,
           cross=cross, cross_interval=cross_interval, spar_method=spar_reg, spar_reg=spar_param)

def eval_cifar10(model, batch_size=128):
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to("cuda"), targets.to("cuda")
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test acc=%.4f" % (test_loss / (num_val_steps), val_acc))

    return val_acc

def train_imagenet(model, epochs=100, batch_size=128, lr=0.01, reg=5e-4,
                  checkpoint_path = '', spar_reg = None, spar_param = 0.0,
                  scheduler='step', finetune=False, cross=False, cross_interval=5,
                  data_dir="../../ILSVRC/Data/CLS-LOC/", distributed=True, init=False):

    if init:
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:32578', rank=0, world_size=1)

    model.cuda()
    # DistributedDataParallel will divide and allocate batch_size to all
    # available GPUs if device_ids are not set
    model = torch.nn.parallel.DistributedDataParallel(model)

    print("Loading Data...")
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(valdir,
    transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]))

    train_dataset = datasets.ImageFolder(traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)


    testloader = torch.utils.data.DataLoader(
        val_dataset,batch_size=256, shuffle=False, sampler=val_sampler,
        num_workers=16, pin_memory=True, collate_fn=fast_collate)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
        num_workers=32, pin_memory=True, collate_fn=fast_collate)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=reg, nesterov=True)

    if scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    elif scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if checkpoint_path != '':
        _load_checkpoint(model, optimizer, checkpoint_path, scheduler)

    _train(model, trainloader, testloader, optimizer, epochs,
           scheduler, checkpoint_path, finetune=finetune,
           cross=cross, cross_interval=cross_interval,
           spar_method=spar_reg, spar_reg=spar_param,
           sampler=train_sampler)

def val_imagenet(model, valdir="../../ILSVRC/Data/CLS-LOC/val/"):
    print("Loading Validation Data...")

    val_dataset = datasets.ImageFolder(valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]))

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,batch_size=256, shuffle=False, sampler=val_sampler,
        num_workers= 16, pin_memory=True, collate_fn=fast_collate)

    return _eval(model, val_loader)

def _train(model, trainloader, testloader,  optimizer, epochs, scheduler=None,
           checkpoint_path='', save_interval=2, device='cuda', finetune=False,
           cross=False, cross_interval=5, spar_method=None, spar_reg = 0.0, sampler=None, ):

    start_epoch, best_acc = _load_checkpoint(model, optimizer, checkpoint_path, scheduler)
    best_acc_path = ''

    if checkpoint_path=='':
        checkpoint_path = os.path.join(os.path.curdir, 'ckpt/'+ model.__class__.__name__
                                       + time.strftime('%m%d%H%M', time.localtime()))
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    log_dir = os.path.join(checkpoint_path, 'log')
    writer = SummaryWriter(log_dir)

    criterion = nn.CrossEntropyLoss().cuda()

    train_coef = True
    for _, m in model.named_modules():
        if isinstance(m, DecomposedConv2D):
            m.coefs.requires_grad = True
            m.basis.requires_grad = True


    end = time.time()
    for epoch in range(start_epoch, epochs):

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(trainloader),
            [batch_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        model.train()

        if sampler is not None:
            sampler.set_epoch(epoch)
        #Swap training basis or coefficient

        if cross and (epoch+1)%cross_interval==0:      #Interleave training bases and coefficient
            train_coef = not train_coef
            print('Swaping Bases and Coefficient Training...')
            for _, m in model.named_modules():
                if isinstance(m, DecomposedConv2D):
                    m.coefs.requires_grad = train_coef
                    m.basis.requires_grad = not train_coef
        elif not cross:
            for _, m in model.named_modules():
                if isinstance(m, DecomposedConv2D):
                    m.coefs.requires_grad = True
                    m.basis.requires_grad = True

        prefetcher = data_prefetcher(trainloader)
        inputs, targets = prefetcher.next()
        batch_idx = 0
        while inputs is not None:
            batch_idx += 1
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            #Sparsity Regularization
            if spar_method == 'l1':
                reg_loss = torch.zeros_like(loss).to('cuda')
                for n, m in model.named_parameters():
                    if "coef" in n:
                        reg_loss += torch.sum(torch.abs(m))
                for n, m in model.named_modules():
                    if isinstance(m, nn.Conv2d) and m.weight.shape[2] == 1:     #Prune 1x1 convolution
                        reg_loss += torch.sum(torch.abs(m.weight))

                loss += reg_loss * spar_reg
            elif spar_method == 'naive_l1':
                reg_loss = torch.zeros_like(loss).to('cuda')
                for n, m in model.named_modules():
                    if isinstance(m, nn.Conv2d):
                        reg_loss += torch.sum(torch.abs(m.weight))

                loss += reg_loss * spar_reg

            loss.backward()

            if finetune:
                for n, m in model.named_parameters():
                    if 'coef' in n:
                        m.grad[m==0] = 0
                for n, m in model.named_modules():
                    if isinstance(m, nn.Conv2d):
                        m.weight.grad[m.weight==0] = 0

            optimizer.step()

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 16 == 0:
                n_step = epoch * len(trainloader) + batch_idx
                progress.display(batch_idx)
                writer.add_scalar('Train/Loss', loss.item(), n_step)
                writer.add_scalar('Train/Top1 ACC', top1.avg, n_step)
                writer.add_scalar('Train/Top5 ACC', top5.avg, n_step)
            inputs, targets = prefetcher.next()

        if scheduler is not None:
            scheduler.step()

        val_acc, top5_acc = _eval(model, testloader, device)
        writer.add_scalar('Test/Top1 Acc', val_acc, epoch)
        writer.add_scalar('Test/Top5 Acc', top5_acc, epoch)

        sparse = 0
        total = 0
        if spar_method == 'l1':
            for n, m in model.named_parameters():
                if 'coef' in n:
                    sparse += np.prod(list(m[m<=1e-7].shape))
                    total += np.prod(list(m.shape))
            print("Sparsity Level %.3f" % (sparse/total))
        elif spar_method == 'naive_l1':
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    sparse += np.prod(list(m.weight[m.weight<=1e-7].shape))
                    total += np.prod(list(m.weight.shape))
            print("Sparsity Level %.3f" % (sparse/total))

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving Weight...")
            if os.path.exists(best_acc_path):
                os.remove(best_acc_path)
            best_acc_path = os.path.join(checkpoint_path, "retrain_weight_%d_%.2f.pt"%(epoch, best_acc))
            torch.save(model.state_dict(), best_acc_path)

        if (epoch+1) % save_interval == 0:
            _save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_path, scheduler)
        torch.cuda.empty_cache()

    return best_acc

def _eval(model, testloader, device='cuda'):
    criterion = nn.CrossEntropyLoss().cuda()
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(testloader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    with torch.no_grad():
        prefetcher = data_prefetcher(testloader)
        inputs, targets = prefetcher.next()
        batch_idx = 0
        end = time.time()
        while inputs is not None:
            batch_idx += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), targets.size(0))
            top1.update(acc1[0], targets.size(0))
            top5.update(acc5[0], targets.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            inputs, targets = prefetcher.next()
            if batch_idx % 10 == 0:
                progress.display(batch_idx)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

#From NVIDIA's Apex Library
def fast_collate(batch, memory_format=torch.contiguous_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

def _save_checkpoint(model, optimizer, cur_epoch, best_acc, save_root, scheduler=None):
    ckpt = {'weight':model.state_dict(),
            'optim': optimizer.state_dict(),
            'cur_epoch':cur_epoch,
            'best_acc':best_acc}
    if scheduler is not None:
        ckpt['scheduler_dict'] = scheduler.state_dict()
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    save_path = os.path.join(save_root, "checkpoint_%d.ckpt"%cur_epoch)
    torch.save(ckpt, save_path)
    print("\033[36mCheckpoint Saved @%d epochs to %s\033[0m"%(cur_epoch+1, save_path))

def _load_checkpoint(model, optimizer, ckpt_path, scheduler=None):
    if not os.path.exists(ckpt_path):
        print("\033[31mCannot find checkpoint folder!\033[0m")
        print("\033[33mTrain From scratch!\033[0m")
        return 0, 0     #Start Epoch, Best Acc
    ckpt_list = os.listdir(ckpt_path)
    last_epoch = -1
    for ckpt_name in ckpt_list:
        if "checkpoint_" in ckpt_name:
            ckpt_epoch = int(ckpt_name.split(".")[0].split('_')[1])
            if ckpt_epoch>last_epoch:
                last_epoch = ckpt_epoch
    if last_epoch == -1:
        print("\033[33mNo checkpoint found!")
        print("Train From scratch!\033[0m")
        return 0, 0
    ckpt_file = os.path.join(ckpt_path, "checkpoint_%d.ckpt"%last_epoch)
    ckpt = torch.load(ckpt_file)
    print("\033[36mStarting from %d epoch.\033[0m"%(ckpt['cur_epoch']))
    model.train()       #This is important for BN
    model.load_state_dict(ckpt['weight'])
    optimizer.load_state_dict(ckpt['optim'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_dict'])

    return ckpt['cur_epoch'], ckpt['best_acc']



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_sparsity(model):
    n_zero = 0
    n_coefs = 0
    n_param = 0
    for n, m in model.named_parameters():
        if 'coef' in n:
            n_zero += np.prod(list(m[m==0].shape))
            n_coefs += np.prod(list(m.shape))

    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            n_param += np.prod(list(m.weight.shape))
            n_zero += np.prod(list(m.weight[m.weight==0].shape))
    n_param += n_coefs
    if n_coefs==0:
        print("No coefficients found!")
        n_coefs = n_param
    print("# of zero parameters: ", n_zero)
    print("# of coefficients: ", n_coefs)
    print("# of total parameters: ", n_param)
    print("Coefficient Sparsity: ", n_zero/n_coefs)
    print("Parameter Sparsity: ", n_zero/n_param)
    return n_zero/n_param, n_zero/n_coefs

def prune_by_std(model, obj='coef', s=1.0):
    with torch.no_grad():
        for n, m in model.named_parameters():
            if obj in n:
                thresh = np.std(m.detach().cpu().numpy()) * s
                m[torch.abs(m)<thresh] = 0
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.weight.shape[2] == 1:     #Prune 1x1 convolution
                thresh = np.std(m.weight.detach().cpu().numpy()) * s
                m.weight[torch.abs(m.weight)<thresh] = 0

def show_basis_angle(params, visualize = False):
    '''
    :param params: Basis, 3d array in shape (layer, num_base, base_dim)
                    Or a list of 2d matrix in shape (num_base, base_dim)
    :return: Cosine Similarity of each layer's bases
    '''
    if isinstance(params, list):
        num_layers = len(params)
        num_basis = [x.shape[0] for x in params]
    else:
        num_layers = params.shape[0]
        num_basis = [params.shape[1]] * num_layers

    layer_sim = []
    for ldx in range(num_layers):
        layer_sim.append(np.zeros((num_basis[ldx], num_basis[ldx])))
        for i in range(5):
            for j in range(5):
                layer_sim[ldx][i, j] = np.dot(params[ldx, i,:].flatten(), params[ldx, j, :].flatten()) / \
                                       np.sqrt(np.sum(params[ldx, i, :] ** 2)*np.sum(params[ldx, j, :] ** 2))
        if visualize:
            plt.matshow(layer_sim[ldx].squeeze())
            plt.title("Layer %d"%ldx)

    if isinstance(params, np.ndarray):
        layer_sim = np.array(layer_sim)

    return layer_sim

#For Sequential Models like VGG16 and AlexNet, and ResNet56
def shrink(model, iterative=True, Linear_req = True):
    model.eval()            #Avoid Problem for BN, just for quick test and need revise
    in_remain = []
    in_redun = []
    out_redun =[]
    out_remain = []
    basis_remain = []
    cors_coef = []
    shortcut_path = [0]
    shortcut_padding = []

    for n, m in model.named_modules():
        #print(n)
        if isinstance(m, DecomposedConv2D):
            temp = m.coefs.data.detach().cpu().numpy().reshape((m.out_channels, m.in_channels, m.num_basis))
            cors_coef.append(temp)
            in_redun.append(np.nonzero(np.sum(np.sum(temp != 0, axis=0), axis=1) == 0)[0])
            out_redun.append(np.nonzero(np.sum(np.sum(temp != 0, axis=1), axis=1) == 0)[0])
            in_remain.append(np.nonzero(np.sum(np.sum(temp != 0, axis=0), axis=1) != 0)[0])
            out_remain.append(np.nonzero(np.sum(np.sum(temp != 0, axis=1), axis=1) != 0)[0])

            basis_remain.append(np.nonzero(np.sum(np.sum(temp != 0, axis=0), axis=0) != 0)[0])
        if 'shortcut' in n:
            shortcut_path.append(len(in_redun)-1)     #Indicate the end point indices of the current shorcut path
            if isinstance(m, LambdaLayer):
                shortcut_padding.append(m)

    #Get the intersection of the input and output channels
    for i in range(1, len(in_remain)):
        out_remain[i-1] = list(set(out_remain[i-1]).intersection(set(in_remain[i])))
        in_remain[i] = list(set(in_remain[i]).intersection(set(out_remain[i-1])))
        out_redun[i-1] = list(set(out_redun[i-1]).union(set(in_redun[i])))
        in_redun[i] = list(set(in_redun[i]).union(set(out_redun[i-1])))

    #Adjust for Skip Connection
    if shortcut_path!=[0]:
        for j in range(1, len(shortcut_path)):
            for i in range(1, len(shortcut_path)):
                start_point = shortcut_path[i-1]
                end_point = shortcut_path[i]

                if end_point in shortcut_padding:       #Avoid Padding Issue
                    continue

                #DEBUG
                #print("ShortCut: ", start_point, "-->", end_point)

                joint_remain = list(set(out_remain[end_point]).union(set(out_remain[start_point])))

                if end_point < len(in_remain) - 1:  #Not the last Layer
                    in_remain[end_point+1] = joint_remain
                out_remain[end_point] = joint_remain
                out_remain[start_point] = joint_remain
                in_remain[start_point+1] = joint_remain

                joint_redun = list(set(out_redun[end_point]).intersection(set(out_redun[start_point])))
                if end_point < len(in_redun) - 1: #Not the last Layer
                    in_redun[end_point+1] = joint_redun
                out_redun[end_point] = joint_redun
                out_redun[start_point] = joint_redun
                in_redun[start_point+1] = joint_redun


    if iterative:
        modify = True

        while modify:
            modify=False
            for i in range(len(in_remain)):
                if i in shortcut_padding or i + 2 in shortcut_padding:
                    continue
                if len(in_redun[i]) != 0:
                    if np.sum(cors_coef[i][:, in_redun[i], :]) != 0:
                        cors_coef[i][:, in_redun[i], :] = 0
                        modify = True

                if len(out_redun[i]) != 0:
                    #print(out_redun[i])
                    if np.sum(cors_coef[i][out_redun[i],:,:]) != 0:
                        cors_coef[i][out_redun[i], :, :] = 0
                        modify = True
                if modify:
                    in_redun[i] = np.nonzero(np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=1) == 0)[0]
                    out_redun[i] = np.nonzero(np.sum(np.sum(cors_coef[i] != 0, axis=1), axis=1) == 0)[0]
                    in_remain[i] = np.nonzero(np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=1) != 0)[0]
                    out_remain[i] = np.nonzero(np.sum(np.sum(cors_coef[i] != 0, axis=1), axis=1) != 0)[0]

                    basis_remain[i] = np.nonzero(np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=0) != 0)[0]
                #print(modify)

            for i in range(1, len(in_remain)):
                    out_remain[i-1] = list(set(out_remain[i-1]).intersection(set(in_remain[i])))
                    in_remain[i] = list(set(in_remain[i]).intersection(set(out_remain[i-1])))
                    out_redun[i-1] = list(set(out_redun[i-1]).union(set(in_redun[i])))
                    in_redun[i] = list(set(in_redun[i]).union(set(out_redun[i-1])))

            # Adjust for Skip Connection
            # Adjust for Skip Connection
            if shortcut_path != [0]:
                for j in range(1, len(shortcut_path)):
                    for i in range(1, len(shortcut_path)):
                        start_point = shortcut_path[i - 1]
                        end_point = shortcut_path[i]

                        if end_point in shortcut_padding:  # Avoid Padding Issue
                            continue

                        # DEBUG
                        # print("ShortCut: ", start_point, "-->", end_point)

                        joint_remain = list(set(out_remain[end_point]).union(set(out_remain[start_point])))

                        if end_point < len(in_remain) - 1:  # Not the last Layer
                            in_remain[end_point + 1] = joint_remain
                        out_remain[end_point] = joint_remain
                        out_remain[start_point] = joint_remain
                        in_remain[start_point + 1] = joint_remain

                        joint_redun = list(set(out_redun[end_point]).intersection(set(out_redun[start_point])))
                        if end_point < len(in_redun) - 1:  # Not the last Layer
                            in_redun[end_point + 1] = joint_redun
                        out_redun[end_point] = joint_redun
                        out_redun[start_point] = joint_redun
                        in_redun[start_point + 1] = joint_redun


    #Construct New Model
    new_model = copy.deepcopy(model)

    i = 0

    for n, m in new_model.named_modules():
        if isinstance(m, DecomposedConv2D):

            skip=False
            if i in shortcut_padding or i + 2 in shortcut_padding:
                print("Skip the border of the layer.")
                skip=True

            print("Shrinking The Layer:", n)

            ori_basis = m.basis.detach().cpu().numpy()
            ori_coefs = m.coefs.detach().cpu().numpy().reshape((m.out_channels, m.in_channels, m.num_basis))
            m.in_channels = len(in_remain[i])
            m.out_channels = m.out_channels if skip else len(out_remain[i])
            m.num_basis = len(basis_remain[i])

            print(i, " |----->New input Channel", m.in_channels, " | New Output Channel", m.out_channels, " | New Basis", m.num_basis)

            r_basis_idx = basis_remain[i]
            r_in_channels_idx = in_remain[i]
            r_out_channels_idx = np.arange(m.out_channels) if skip else out_remain[i]

            new_basis = np.zeros((m.num_basis, m.kernel_size[0]*m.kernel_size[1]), dtype=np.float32)
            new_coefs = np.zeros((m.out_channels, m.in_channels, m.num_basis),  dtype=np.float32)

            for bidx in range(m.num_basis):
                new_basis[bidx, :] = ori_basis[r_basis_idx[bidx], :]

            for cin in range(m.in_channels):
                for cout in range(m.out_channels):
                    #print(ori_coefs.shape)
                    #print(new_coefs.shape)
                    new_coefs[cout, cin, :] = ori_coefs[r_out_channels_idx[cout], r_in_channels_idx[cin], r_basis_idx]

            m.coefs = nn.Parameter(torch.tensor(new_coefs.reshape(m.out_channels*m.in_channels, m.num_basis)), requires_grad=True)
            m.basis = nn.Parameter(torch.tensor(new_basis), requires_grad=False)
            i += 1

        if isinstance(m, torch.nn.BatchNorm2d):
            print("Shrinking The BN Layer:", n)
            m.num_features = len(out_remain[i-1])
            print("----->New number of features: ", m.num_features)
            ori_weight = m.weight[list(out_remain[i-1])].detach().cpu().numpy()
            ori_bias = m.bias[list(out_remain[i-1])].detach().cpu().numpy()
            ori_mean = m.running_mean[list(out_remain[i-1])].detach().cpu().numpy()
            ori_var = m.running_var[list(out_remain[i-1])].detach().cpu().numpy()

            m.weight = nn.Parameter(torch.tensor(ori_weight))
            m.bias = nn.Parameter(torch.tensor(ori_bias))
            m.running_mean = torch.tensor(ori_mean)
            m.running_var = torch.tensor(ori_var)
            
        if isinstance(m, torch.nn.Linear) and Linear_req:
            print("Shrinking The First Linear Layer:", n)
            m.in_features = len(out_remain[i-1])
            ori_weight = m.weight.detach().cpu().numpy()
            new_weight = ori_weight[:, out_remain[i-1]]
            m.weight = nn.Parameter(torch.Tensor(new_weight))
            Linear_req=False

    return new_model, [len(x) for x in out_remain]


#Temperary implementation of shrinking function for ResNet50
#More redundancy could be explored if we use padding scheme or something else with non-downsampling connection
def shrink_resnet(model, iterative):
    model.eval()  # Avoid Problem for BN, just for quick test and need revise

    in_remain = []
    in_redun = []
    out_redun = []
    out_remain = []
    basis_remain = []
    cors_coef = []

    sc_in_remain = []
    sc_in_redun = []
    sc_out_redun = []
    sc_out_remain = []
    sc_cors_coef = []

    shortcut_path = [0]

    #Derive Redundancy
    for n, m in model.named_modules():
        # print(n)
        if isinstance(m, DecomposedConv2D) or (isinstance(m, nn.Conv2d) and m.weight.shape[2] == 1):
            if isinstance(m, DecomposedConv2D):
                temp = m.coefs.data.detach().cpu().numpy().reshape((m.out_channels, m.in_channels, m.num_basis))
            else:   #1x1 Conv
                temp = m.weight.data.detach().cpu().numpy().reshape((m.out_channels, m.in_channels, 1))

            if 'downsample' in n:
                shortcut_path.append(len(out_remain) - 1)
                sc_cors_coef.append(temp)
                sc_in_redun.append(np.nonzero(np.sum(np.sum(temp != 0, axis=0), axis=1) == 0)[0])
                sc_out_redun.append(np.nonzero(np.sum(np.sum(temp != 0, axis=1), axis=1) == 0)[0])
                sc_in_remain.append(np.nonzero(np.sum(np.sum(temp != 0, axis=0), axis=1) != 0)[0])
                sc_out_remain.append(np.nonzero(np.sum(np.sum(temp != 0, axis=1), axis=1) != 0)[0])
            else:
                cors_coef.append(temp)
                #Specially tailored for BottleNeck Structure
                if ".conv1" in n:   #Do not modify the input of the bottleneck
                    in_redun.append([])
                    in_remain.append(np.arange(m.in_channels))
                else:
                    in_redun.append(np.nonzero(np.sum(np.sum(temp != 0, axis=0), axis=1) == 0)[0])
                    in_remain.append(np.nonzero(np.sum(np.sum(temp != 0, axis=0), axis=1) != 0)[0])

                if ".conv3" in n:
                    out_redun.append([])
                    out_remain.append(np.arange(m.out_channels,))
                else:
                    out_redun.append(np.nonzero(np.sum(np.sum(temp != 0, axis=1), axis=1) == 0)[0])
                    out_remain.append(np.nonzero(np.sum(np.sum(temp != 0, axis=1), axis=1) != 0)[0])

                basis_remain.append(np.nonzero(np.sum(np.sum(temp != 0, axis=0), axis=0) != 0)[0])

    # Adjust for Skip Connection
    if shortcut_path != [0]:
        for i in range(1, len(shortcut_path)):
            start_point = shortcut_path[i - 1]
            end_point = shortcut_path[i]

            # DEBUG
            print("ShortCut: ", start_point, "-->", end_point)
            joint_remain_start = list(set(out_remain[start_point]).union(set(sc_in_remain[i-1])))
            out_remain[start_point] = joint_remain_start
            sc_in_remain[i-1] = joint_remain_start
            joint_redun_start = list(set(out_redun[start_point]).intersection(set(sc_in_redun[i-1])))
            out_redun[start_point] = joint_redun_start
            sc_in_redun[i-1] = joint_redun_start

            joint_remain_end = list(set(out_remain[end_point]).union(set(sc_out_remain[i-1])))
            out_remain[end_point] = joint_remain_end
            sc_out_remain[i-1] = joint_remain_end
            joint_redun_end = list(set(out_redun[end_point]).intersection(set(sc_out_redun[i-1])))
            out_redun[end_point] = joint_redun_end
            sc_out_redun[i-1] = joint_redun_end



    # Get the intersection of the input and output channels
    for i in range(1, len(in_remain)):
        out_remain[i - 1] = list(set(out_remain[i - 1]).intersection(set(in_remain[i])))
        in_remain[i] = list(set(in_remain[i]).intersection(set(out_remain[i - 1])))
        out_redun[i - 1] = list(set(out_redun[i - 1]).union(set(in_redun[i])))
        in_redun[i] = list(set(in_redun[i]).union(set(out_redun[i - 1])))

    if shortcut_path != [0]:
        for i in range(1, len(shortcut_path)):
            start_point = shortcut_path[i - 1]
            end_point = shortcut_path[i]

            sc_in_remain[i - 1] = out_remain[start_point]
            sc_in_redun[i - 1] = out_redun[start_point]

            sc_out_remain[i - 1] = out_remain[end_point]
            sc_out_redun[i - 1] = out_redun[end_point]


    if iterative:
        modify = True
        while modify:
            modify = False
            for i in range(len(in_remain)):
                if len(in_redun[i]) != 0:
                    if np.sum(cors_coef[i][:, in_redun[i], :]) != 0:
                        cors_coef[i][:, in_redun[i], :] = 0
                        in_redun[i] = np.nonzero(np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=1) == 0)[0]
                        in_remain[i] = np.nonzero(np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=1) != 0)[0]
                        modify = True

                if len(out_redun[i]) != 0:
                    # print(out_redun[i])
                    if np.sum(cors_coef[i][out_redun[i], :, :]) != 0:
                        cors_coef[i][out_redun[i], :, :] = 0
                        out_redun[i] = np.nonzero(np.sum(np.sum(cors_coef[i] != 0, axis=1), axis=1) == 0)[0]
                        out_remain[i] = np.nonzero(np.sum(np.sum(cors_coef[i] != 0, axis=1), axis=1) != 0)[0]
                        modify = True

                if modify:
                    basis_remain[i] = np.nonzero(np.sum(np.sum(cors_coef[i] != 0, axis=0), axis=0) != 0)[0]
                # print(modify)
            for i in range(len(sc_in_remain)):
                if len(sc_in_redun[i]) != 0:
                    if np.sum(sc_cors_coef[i][:, sc_in_redun[i], :]) != 0:
                        sc_cors_coef[i][:, sc_in_redun[i], :] = 0
                        modify = True

                if len(sc_out_redun[i]) != 0:
                    # print(out_redun[i])
                    if np.sum(sc_cors_coef[i][sc_out_redun[i], :, :]) != 0:
                        sc_cors_coef[i][sc_out_redun[i], :, :] = 0
                        modify = True

                if modify:
                    sc_in_redun[i] = (np.nonzero(np.sum(np.sum(sc_cors_coef[i] != 0, axis=0), axis=1) == 0)[0])
                    sc_out_redun[i] = (np.nonzero(np.sum(np.sum(sc_cors_coef[i] != 0, axis=1), axis=1) == 0)[0])
                    sc_in_remain[i] = (np.nonzero(np.sum(np.sum(sc_cors_coef[i] != 0, axis=0), axis=1) != 0)[0])
                    sc_out_remain[i] = (np.nonzero(np.sum(np.sum(sc_cors_coef[i] != 0, axis=1), axis=1) != 0)[0])

            # Adjust for Skip Connection
            if shortcut_path != [0]:
                for i in range(1, len(shortcut_path)):
                    start_point = shortcut_path[i - 1]
                    end_point = shortcut_path[i]

                    # DEBUG
                    # print("ShortCut: ", start_point, "-->", end_point)
                    joint_remain_start = list(set(out_remain[start_point]).union(set(sc_in_remain[i - 1])))
                    out_remain[start_point] = joint_remain_start
                    sc_in_remain[i - 1] = joint_remain_start
                    joint_redun_start = list(set(out_redun[start_point]).intersection(set(sc_in_redun[i - 1])))
                    out_redun[start_point] = joint_redun_start
                    sc_in_redun[i - 1] = joint_redun_start

                    joint_remain_end = list(set(out_remain[end_point]).union(set(sc_out_remain[i - 1])))
                    out_remain[end_point] = joint_remain_end
                    sc_out_remain[i - 1] = joint_remain_end
                    joint_redun_end = list(set(out_redun[end_point]).intersection(set(sc_out_redun[i - 1])))
                    out_redun[end_point] = joint_redun_end
                    sc_out_redun[i - 1] = joint_redun_end

            # Get the intersection of the input and output channels
            for i in range(1, len(in_remain)):
                out_remain[i - 1] = list(set(out_remain[i - 1]).intersection(set(in_remain[i])))
                in_remain[i] = list(set(in_remain[i]).intersection(set(out_remain[i - 1])))
                out_redun[i - 1] = list(set(out_redun[i - 1]).union(set(in_redun[i])))
                in_redun[i] = list(set(in_redun[i]).union(set(out_redun[i - 1])))

            if shortcut_path != [0]:
                for i in range(1, len(shortcut_path)):
                    start_point = shortcut_path[i - 1]
                    end_point = shortcut_path[i]

                    sc_in_remain[i - 1] = out_remain[start_point]
                    sc_in_redun[i - 1] = out_redun[start_point]

                    sc_out_remain[i - 1] = out_remain[end_point]
                    sc_out_redun[i - 1] = out_redun[end_point]

    # Construct New Model
    new_model = copy.deepcopy(model)

    new_width = []

    i = 0
    sc = 0
    Linear_req = True
    for n, m in new_model.named_modules():
        if isinstance(m, DecomposedConv2D) or (isinstance(m, nn.Conv2d) and m.weight.shape[2]==1):
            #if isinstance(m, DecomposedConv2D):
            new_width.append(m.out_channels)

            if isinstance(m, nn.Conv2d):
                if 'downsample' in n:
                    print("Shrinking The Shortcut:", n)
                else:
                    print("Shrinking The Layer:", n)
                ori_weight = m.weight.detach().cpu().numpy().reshape((m.out_channels, m.in_channels, 1, 1))
                m.in_channels = len(sc_in_remain[sc]) if 'downsample' in n else len(in_remain[i])
                m.out_channels = len(sc_out_remain[sc]) if 'downsample' in n else len(out_remain[i])

                print("----->New input Channel", m.in_channels, " | New Output Channel", m.out_channels)

                r_in_channels_idx = sc_in_remain[sc] if 'downsample' in n else in_remain[i]
                r_out_channels_idx = sc_out_remain[sc] if 'downsample' in n else out_remain[i]

                new_weight = np.zeros((m.out_channels, m.in_channels, 1, 1), dtype=np.float32)

                for cin in range(m.in_channels):
                    for cout in range(m.out_channels):
                        new_weight[cout, cin, :, :] = ori_weight[r_out_channels_idx[cout], r_in_channels_idx[cin], :, :]

                m.weight = nn.Parameter(torch.tensor(new_weight), requires_grad=True)
                if 'downsample' in n:
                    sc += 1
                else:
                    i += 1

            else:
                print("Shrinking The Layer:", n)

                ori_basis = m.basis.detach().cpu().numpy()
                ori_coefs = m.coefs.detach().cpu().numpy().reshape((m.out_channels, m.in_channels, m.num_basis))
                m.in_channels = len(in_remain[i])
                m.out_channels = len(out_remain[i])
                m.num_basis = len(basis_remain[i])

                print("----->New input Channel", m.in_channels, " | New Output Channel", m.out_channels, " | New Basis",
                      m.num_basis)

                r_basis_idx = basis_remain[i]
                r_in_channels_idx = in_remain[i]
                r_out_channels_idx = out_remain[i]

                new_basis = np.zeros((m.num_basis, m.kernel_size[0] * m.kernel_size[1]), dtype=np.float32)
                new_coefs = np.zeros((m.out_channels, m.in_channels, m.num_basis), dtype=np.float32)

                for bidx in range(m.num_basis):
                    new_basis[bidx, :] = ori_basis[r_basis_idx[bidx], :]

                for cin in range(m.in_channels):
                    for cout in range(m.out_channels):
                        # print(ori_coefs.shape)
                        # print(new_coefs.shape)
                        new_coefs[cout, cin, :] = ori_coefs[r_out_channels_idx[cout], r_in_channels_idx[cin], r_basis_idx]

                m.coefs = nn.Parameter(torch.tensor(new_coefs.reshape(m.out_channels * m.in_channels, m.num_basis)),
                                       requires_grad=True)
                m.basis = nn.Parameter(torch.tensor(new_basis), requires_grad=False)
                i += 1

        if isinstance(m, torch.nn.BatchNorm2d):
            print("Shrinking The BN Layer:", n)
            m.num_features = len(out_remain[i - 1])
            print("----->New number of features: ", m.num_features)
            ori_weight = m.weight[list(out_remain[i - 1])].detach().cpu().numpy()
            ori_bias = m.bias[list(out_remain[i - 1])].detach().cpu().numpy()
            ori_mean = m.running_mean[list(out_remain[i - 1])].detach().cpu().numpy()
            ori_var = m.running_var[list(out_remain[i - 1])].detach().cpu().numpy()

            m.weight = nn.Parameter(torch.tensor(ori_weight))
            m.bias = nn.Parameter(torch.tensor(ori_bias))
            m.running_mean = torch.tensor(ori_mean)
            m.running_var = torch.tensor(ori_var)

        if isinstance(m, torch.nn.Linear) and Linear_req:
            print("Shrinking The First Linear Layer:", n)
            m.in_features = len(out_remain[i - 1])
            ori_weight = m.weight.detach().cpu().numpy()
            new_weight = ori_weight[:, out_remain[i - 1]]
            m.weight = nn.Parameter(torch.Tensor(new_weight))
            Linear_req = False

    return new_model, new_width
