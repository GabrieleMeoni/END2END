from __future__ import print_function, division
import os

import torch

from utils import net_builder
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader
from sklearn.metrics import confusion_matrix
from utils import get_classes_name
from train_utils import mcc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path", type=str, default="./saved_models/fixmatch/model_best.pth"
    )
    parser.add_argument("--use_train_model", action="store_true")
    parser.add_argument("--export_confusion_matrix", action="store_true")
    parser.add_argument("--plot_confusion_matrix", action="store_true")
    parser.add_argument("--csv_path", type=str, default=".")

    """
    Backbone Net Configurations
    """
    parser.add_argument("--net", type=str, default="WideResNet")
    parser.add_argument("--net_from_name", type=bool, default=False)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--leaky_slope", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)

    """
    Data Configurations
    """
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training. "
    )
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = (
        checkpoint["train_model"] if args.use_train_model else checkpoint["eval_model"]
    )

    _net_builder = net_builder(
        args.net,
        args.net_from_name,
        {
            "depth": args.depth,
            "widen_factor": args.widen_factor,
            "leaky_slope": args.leaky_slope,
            "dropRate": args.dropout,
        },
    )

    _eval_dset = SSL_Dataset(
        name=args.dataset, train=False, data_dir=args.data_dir, seed=args.seed
    )

    eval_dset_basic = _eval_dset.get_dset()
    args.num_classes = _eval_dset.num_classes
    args.num_channels = _eval_dset.num_channels

    net = _net_builder(num_classes=args.num_classes, in_channels=args.channels)
    net.load_state_dict(load_model)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    eval_loader = get_data_loader(eval_dset_basic, args.batch_size, num_workers=1)

    acc = 0.0
    y_true = []
    y_pred = []
    n = 0
    with torch.no_grad():
        for image, target in eval_loader:
            image = image.type(torch.FloatTensor).cuda()
            logit = net(image)
            y_pred += list(logit.cpu().max(1)[1])
            y_true += list(target)
            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()

            if n == 0:
                pred = logit
                correct = target
                n += 1
            else:
                pred = torch.cat((pred, logit), axis=0)

                correct = torch.cat((correct, target), axis=0)
    print(f"Test accuracy: {acc / len(eval_dset_basic)} Test mcc: {mcc(pred, correct)}")

    if args.export_confusion_matrix:
        print("\nExtracting the confusion matrix...")

        class_names = sorted(get_classes_name(_eval_dset))
        c_m = confusion_matrix(y_true, y_pred)
        print(class_names)
        print(c_m)
        c_m_names = pd.DataFrame(c_m, columns=class_names, index=class_names)

        class_dict = {i: class_names[i] for i in range(len(c_m_names))}
        class_predicted_list = []
        class_expected_list = []
        for i in range(len(y_true)):
            class_predicted_list.append(class_dict[int(y_pred[i])])
            class_expected_list.append(class_dict[int(y_true[i])])
        classification_data = {
            "y_true": class_expected_list,
            "Y_pred": class_predicted_list,
        }
        class_df = pd.DataFrame(classification_data)

        if args.load_path[-1] == os.path.sep:
            args.load_path = args.load_path[:-1]

        csv_name = args.load_path.split(os.path.sep)[-2]

        if args.plot_confusion_matrix:
            plt.figure(figsize=(8, 8), dpi=100)
            # Scale up the size of all text
            sns.set(font_scale=1.1)
            ax = sns.heatmap(
                c_m,
                annot=True,
                fmt="d",
            )
            ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
            ax.xaxis.set_ticklabels(class_names)
            ax.set_ylabel("Actual", fontsize=14, labelpad=20)
            ax.yaxis.set_ticklabels(class_names)
            plt.savefig(os.path.join(args.csv_path, csv_name + "_conf_matrix.png"))

        c_m_names.to_csv(os.path.join(args.csv_path, csv_name + "_conf_matrix.csv"))
        class_df.to_csv(os.path.join(args.csv_path, csv_name + "_detailed_results.csv"))
