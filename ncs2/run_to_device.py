import sys
import os

sys.path.insert(1, os.path.join("..", "..", "MSMatch"))
import time
import torch
import argparse
from datasets.ssl_dataset import SSL_Dataset
from termcolor import colored
from train_utils import mcc
from datasets.data_utils import get_data_loader

from openvino import runtime as ov


def get_performance(targets, preds):
    """Extracts accuracy and MCC performance.

    Args:
        targets (list): expected classes.
        preds (list): predicted classes.

    Returns:
        float: model accuracy.
        float: model MCC.
    """
    acc = 0
    n = 0
    for target, pred in zip(targets, preds):
        acc += torch.from_numpy(pred).max(1)[1].eq(target).sum().numpy()

        if n == 0:
            pred_stacked = torch.from_numpy(pred)
            correct = target
        else:
            pred_stacked = torch.cat((pred_stacked, torch.from_numpy(pred)), axis=0)
            correct = torch.cat((correct, target), axis=0)
        n += 1
    return acc / n, mcc(pred_stacked, correct)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--load_path",
        type=str,
        default=r"C:\Users\meoni\Documents\ESA\Projects\END2END\ncs2\openvino\output\efficientnet-lite0.xml",
    )
    parser.add_argument("--test_dataset", type=str, default="thraws_swir_test")
    parser.add_argument(
        "--device",
        type=str,
        default="MYRIAD",
        help="Embedded device. Supported " "CPU" ", " "GPU" ", " "MYRIAD" ".",
    )
    args = parser.parse_args()

    # Create SSL loaders
    data_dir = os.path.join(os.getcwd(), "..", "..", "MSMatch", "DATA")
    print(
        "Import "
        + colored("test", "green")
        + " dataset: "
        + colored(args.test_dataset, "blue")
    )
    _test_dset = SSL_Dataset(name=args.test_dataset, data_dir=data_dir)
    test_dset = _test_dset.get_dset()

    # Compile model
    # Initialize OpenVINO Runtime.

    print("Compiling model: " + colored(args.load_path, "blue"))
    ie_core = ov.Core()
    # Read the network and corresponding weights from a file.
    model = ie_core.read_model(model=args.load_path)
    # Compile the model for CPU (you can choose manually CPU, GPU, MYRIAD etc.)
    # or let the engine choose the best available device (AUTO).
    compiled_model = ie_core.compile_model(model=model, device_name=args.device)

    # Get the input and output nodes.
    _ = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Preparing placeholders for targets and predictions and preparing data for the inference.
    predicted = []
    targets = []
    test_loader = get_data_loader(test_dset, 1, num_workers=1)

    # Starting timing profiling.
    start_time = time.time()

    # Running the inference on the target device.
    print(colored("Start testing...", "red"))
    for X, y in test_loader:
        predicted.append(compiled_model([X])[output_layer])
        targets.append(y)

    # Ending timing profiling.
    stop_time = time.time()
    print(colored("Testing finished.", "green"))
    print(colored("Calculating results...", "blue"))

    # Extracting the inference time
    inference_time = stop_time - start_time
    # Extracting accuracy and MCC performance
    accuracy, mcc_results = get_performance(targets, predicted)

    print("Total inference time[s]: " + colored(str(inference_time), "blue"))
    print("Number of patches: " + colored(str(len(test_loader)), "yellow"))
    print(
        "Average inference time per patch[s]: "
        + colored(str(inference_time / len(test_loader)), "cyan")
    )
    print("Accuracy [%]: " + colored(str(accuracy * 100), "green"))
    print("MCC: " + colored(str(mcc_results), "red"))


if __name__ == "__main__":
    main()
