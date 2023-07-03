import sys
import os

sys.path.insert(1, os.path.join("..", "onboard_prototype"))
import torch
import argparse
from termcolor import colored
from onboard_detector_processor import oboardDetectorProcessor
from onboard_prototype_utils import load_data


def main():
    parser = argparse.ArgumentParser(description="")

    # DETECTOR AND SATELLITE ARGUMENTS
    parser.add_argument("--satellite", type=str)
    parser.add_argument("--detector_number", type=int)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"C:\Users\meoni\Documents\ESA\Projects\END2END\onboard_prototype\my_dir",
    )

    # DETECTOR PROCESSOR ARGUMENTS
    parser.add_argument("--model", type=str, default="efficientnet-lite0")
    parser.add_argument(
        "--load_path",
        type=str,
        default="model_best.pth",
    )

    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--leaky_slope", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)

    # OUTPUT DIRECTORY ARGUMENTS
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()

    # Loading data
    data_loaded = load_data(args.dataset_path)

    detector_number_str = str(args.detector_number)

    # Selecting the event of interests
    event_compatible = None
    for event in data_loaded:
        if (data_loaded[event][0] == args.satellite) and (
            detector_number_str in list(data_loaded[event][1].keys())
        ):
            event_compatible = data_loaded[event]
            break

    if event_compatible is None:
        raise ValueError(
            "Impossible to find an event having satellite: "
            + colored(args.satellite, "red")
            + " and detector number: "
            + colored(detector_number_str)
            + "."
        )

    # Granules numbers and lists
    granules_numbers_list = [
        list(event_compatible[1][detector_number_str].keys())[::-1]
        if not (int(args.detector_number) % 2)
        else list(event_compatible[1][detector_number_str].keys())
    ][0]
    granules_numbers = event_compatible[1][detector_number_str]

    # Create dummy input
    dummy_input = granules_numbers[granules_numbers_list[0]]

    # Load onboard processor detector
    onboard_processor = oboardDetectorProcessor(
        args.satellite,
        args.detector_number,
        model_name=args.model,
        depth=args.depth,
        widen_factor=args.widen_factor,
        leaky_slope=args.leaky_slope,
        dropout=args.dropout,
    )
    print("Loading model: " + colored(args.load_path, "green") + "...")
    onboard_processor.load_model(args.load_path)
    onboard_processor.eval()

    # Export to ONNX
    OUTPUT_DIR = args.output_dir

    onnx_path = os.path.join(
        OUTPUT_DIR,
        "onboard_processor_D" + detector_number_str + "_" + args.model + ".onnx",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Creating an ONNX model from: " + colored(args.load_path, "red"))
    torch.onnx.export(onboard_processor, dummy_input, onnx_path, opset_version=16)

    print("ONNX model exported to: " + colored(onnx_path, "green"))


if __name__ == "__main__":
    main()
