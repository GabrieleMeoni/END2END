import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from fastai.metrics import MatthewsCorrCoef
import math
import os


class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """

    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, tb_dict, it, suffix=None):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ""

        for key, value in tb_dict.items():
            self.writer.add_scalar(suffix + key, value, it)


class AverageMeter(object):
    """
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
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


def get_OPT(
    net,
    name="SGD",
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    bn_wd_skip=True,
):
    """
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    """
    if name == "SGD":
        decay = []
        no_decay = []
        for name, param in net.named_parameters():
            if ("bn" in name) and bn_wd_skip:
                no_decay.append(param)
            else:
                decay.append(param)

        per_param_args = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = torch.optim.SGD(
            per_param_args,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif name == "Adam" or name == "ADAM":
        if lr > 0.005:
            raise ValueError(
                "Learning rate is " + str(lr) + ". That is too high for ADAM."
            )

        decay = []
        no_decay = []
        for name, param in net.named_parameters():
            if ("bn" in name) and bn_wd_skip:
                no_decay.append(param)
            else:
                decay.append(param)

        per_param_args = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = torch.optim.Adam(
            per_param_args,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

    return optimizer


def get_cosine_schedule_with_warmup(
    optimizer,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    num_warmup_steps=0,
    last_epoch=-1,
):
    """
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    """

    def _lr_lambda(current_step):
        """
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        """

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Args
        output: logits or probs (num of batch, num of classes)
        target: (num of batch, 1) or (num of batch, )
        topk: list of returned k

    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    with torch.no_grad():
        maxk = max(topk)  # get k in top-k
        batch_size = target.size(0)  # get batch size of target

        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
        # return: value, index
        _, pred = output.topk(
            k=maxk, dim=1, largest=True, sorted=True
        )  # pred: [num of batch, k]
        pred = pred.t()  # pred: [k, num of batch]

        # [1, num of batch] -> [k, num_of_batch] : bool
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        # np.shape(res): [k, 1]
        return res


def mcc(output, target):
    """
    Computes the Matthews Correlation Coefficient.

    Args
        output: logits or probs (num of batch, num of classes)
        target: (num of batch, 1) or (num of batch, )
    refer: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    """
    mcc = MatthewsCorrCoef()
    with torch.no_grad():
        pred_c = torch.argmax(output, axis=1)
        if len(target.shape) > 1:
            target_c = torch.argmax(target, axis=1)
        else:
            target_c = target
    return mcc(pred_c, target_c)


def ce_loss(logits, targets, weight=None, use_hard_labels=True, reduction="none"):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        weight: weights for loss if hard labels are used.
        use_hard_labels: If True, targets have [Batch size] shape with int values.
                         If False, the target is vector. Default to True.
    """
    if use_hard_labels:
        if weight is not None:
            return F.cross_entropy(
                logits, targets.long(), weight=weight, reduction=reduction
            )
        else:
            return F.cross_entropy(logits, targets.long(), reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
