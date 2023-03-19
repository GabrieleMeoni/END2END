import torch
import nncf  # Important - should be imported directly after torch.
from train_utils import mcc
import time
import torch.optim
import warnings
warnings.filterwarnings("ignore",category=UserWarning)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def validate(eval_loader, model, device):
    acc = 0.0
    y_true=[]
    y_pred=[]
    n=0
    with torch.no_grad():
        for image, target in eval_loader:
            image = image.type(torch.FloatTensor).to(device)
            logit = model(image)
            y_pred+=list(logit.cpu().max(1)[1])
            y_true+=list(target)
            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()

            if n == 0:
                    pred=logit
                    correct=target
            else:
                pred=torch.cat((pred, logit), axis=0)
                correct=torch.cat((correct, target), axis=0)
            
            n+=len(target)

    return acc/n, mcc(pred, correct)

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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / batch_size)))
        return res
    
def mcc_runing(output, target,):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        res = [mcc(output, target)/ batch_size]
  
        return res

def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter("Time", ":3.3f")
    losses = AverageMeter("Loss", ":2.3f")
    top1 = AverageMeter("Acc@1", ":2.2f")
    mcc_metric = AverageMeter("MCC", ":2.2f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, losses, top1, mcc_metric], prefix="Epoch:[{}]".format(epoch)
    )

    # Switch to train mode.
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # Compute output.
        output = model(images)
        loss = criterion(output, target)

        # Measure accuracy and record loss.
        acc1= accuracy(output, target, topk=(1,))
        mcc_k = mcc_runing(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        mcc_metric.update(mcc_k[0], images.size(0))

        # Compute gradient and do opt step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        print_frequency = 50
        if i % print_frequency == 0:
            progress.display(i)