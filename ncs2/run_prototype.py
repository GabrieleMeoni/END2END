import sys
import os

sys.path.insert(1, os.path.join("..", "onboard_prototype"))
import time
import torch
import argparse
from matplotlib.patches import Rectangle
from termcolor import colored
from openvino import runtime as ov
import matplotlib.pyplot as plt
from onboard_detector_processor import oboardDetectorProcessor
from onboard_prototype_utils import load_data, plot
import numpy as np


def plot_results(x_coreg_list, y_list, save_fig_path):
    fig, ax = plt.subplots(3, 2, figsize=(60, 60))

    for n in range(3):
        x = x_coreg_list[n]
        y = y_list[n]
        plot(x, x_max=1, ax=ax[n, 0])
        ax[n, 0].set_title("Image")
        ax[n, 1].set_title("Predicted mask")
        # x_map=torch.zeros([x.shape[1], x.shape[2], 3])
        ax[n, 1].set_xlim(0, x.shape[2])
        ax[n, 1].set_ylim(0, x.shape[1])
        patch_list = []
        for h in range(y.shape[1]):
            for v in range(y.shape[0]):
                if y[v, h] == 0:
                    patch_list.append(
                        Rectangle(
                            (h * 256, x.shape[1] - v * 256), 256, -256, color="red"
                        )
                    )
        for patch in patch_list:
            ax[n, 1].add_patch(patch)
    plt.savefig(save_fig_path)


def main():
    parser = argparse.ArgumentParser(description="")

    # DETECTOR AND SATELLITE ARGUMENTS
    parser.add_argument("--event_name", type=str)
    parser.add_argument("--detector_number", type=int)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"C:\Users\meoni\Documents\ESA\Projects\END2END\onboard_prototype\my_dir",
    )

    # ONNX ARGUMENTS
    parser.add_argument(
        "--ir_path",
        type=str,
        default=r"C:\Users\meoni\Documents\ESA\Projects\END2END\ncs2\output\efficientnet-lite0.xml",
    )

    # OUTPUT DIRECTORY ARGUMENTS
    parser.add_argument("--output_dir", type=str, default="output")

    # EMBEDDED DEVICE ARGUMENTS
    parser.add_argument(
        "--device",
        type=str,
        default="MYRIAD",
        help="Embedded device. Supported " "CPU" ", " "GPU" ", " "MYRIAD" ".",
    )
    parser.add_argument(
        "--onboard_processor_device",
        type=str,
        default="CPU",
        help="Onboard processor device. Supported " "CPU" ", " "GPU" ".",
    )
    args = parser.parse_args()

    if torch.cuda.is_available() and args.onboard_processor_device == "GPU":
        onboard_processor_device = torch.device("cuda")
    else:
        onboard_processor_device = torch.device("cpu")

    # Loading data
    data_loaded = load_data(args.dataset_path)

    # Parsing detector number and covert it to a string
    detector_number_str = str(args.detector_number)

    # Selecting the event of interests
    try:
        event = data_loaded[args.event_name]
    except:  # noqa E722
        raise ValueError(
            "Impossible to find the event: " + colored(args.event_name, "red") + "."
        )

    if not (detector_number_str in list(event[1].keys())):
        raise ValueError(
            "The event: "
            + colored(args.event_name, "red")
            + " has no granule with detector number: "
            + colored(detector_number_str)
            + "."
        )

    # Granules numbers and lists
    granules_numbers_list = [
        list(event[1][detector_number_str].keys())[::-1]
        if not (int(args.detector_number) % 2)
        else list(event[1][detector_number_str].keys())
    ][0]
    granules_dict = event[1][detector_number_str]

    # Load onboard processor detector
    onboard_processor = oboardDetectorProcessor(
        event[0], args.detector_number, model_name=None, device=onboard_processor_device
    )

    # IR path
    ir_path = args.ir_path

    print("Compiling model: " + colored(ir_path, "blue"))
    ie_core = ov.Core()
    # Read the network and corresponding weights from a file.
    model = ie_core.read_model(model=ir_path)
    # Compile the model for CPU (you can choose manually CPU, GPU, MYRIAD etc.)
    # or let the engine choose the best available device (AUTO).
    compiled_model = ie_core.compile_model(model=model, device_name=args.device)

    # Get the input and output nodes.
    _ = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print(colored("Start testing...", "red"))

    # Results
    patch_time_s = 0
    inference_time_s = 0
    data_move_s = 0
    coregistration_time_s = 0
    start_time = time.time()
    masks_list = []
    x_coreg_list = []

    for granule_number in granules_numbers_list:
        start_granule_s = time.time()
        granule = granules_dict[granule_number].to(onboard_processor_device)
        start_coregistration_s = time.time()
        data_move_s += start_coregistration_s - start_granule_s
        granule_coreg = onboard_processor.coarse_coregistration(granule / 255)
        coregistration_time_s += time.time() - start_coregistration_s
        onboard_processor.init_patch_engine(granule_coreg)

        # Size of the output mask
        n_v_max, n_h_max = onboard_processor.get_output_shape()

        y_demosaicked = np.zeros([n_v_max, n_h_max])
        n_v, n_h = 0, 0

        while not (onboard_processor.processing_finished()):
            patch_time_start_s = time.time()
            granule_patch = onboard_processor.get_next_patch().to(torch.device("cpu"))
            timestamp_s = time.time()
            patch_time_s += timestamp_s - patch_time_start_s
            y_demosaicked[n_v, n_h] = compiled_model([granule_patch.unsqueeze(0)])[
                output_layer
            ].argmax()
            inference_time_s += time.time() - timestamp_s
            if n_v < n_v_max - 1:
                n_v += 1
            else:
                n_v = 0
                n_h += 1

        masks_list.append(y_demosaicked)
        x_coreg_list.append(granule_coreg.to(torch.device("cpu")))

    stop_time = time.time()
    N_processed = len(granules_numbers_list)
    processing_time_s = stop_time - start_time

    print(colored("Testing finished.", "green"))
    print(colored("Calculating results...", "blue"))

    print("Total inference time[s]: " + colored(str(processing_time_s), "blue"))
    print("Number of granules: " + colored(str(N_processed), "yellow"))
    print(
        "Average total time to move data per granule[s]: "
        + colored(str(data_move_s / N_processed), "cyan")
    )
    print(
        "Average total patch time per granule[s]: "
        + colored(str(patch_time_s / N_processed), "red")
    )
    print(
        "Average total coregistration time per granule[s]: "
        + colored(str(coregistration_time_s / N_processed), "blue")
    )
    print(
        "Average inference time per granule[s]: "
        + colored(str(inference_time_s / N_processed), "green")
    )
    print(
        "Average total processing time per granule[s]: "
        + colored(str(processing_time_s / N_processed), "cyan")
    )

    plot_results(
        x_coreg_list,
        masks_list,
        args.event_name + "_" + str(args.detector_number) + "_mask.png",
    )


if __name__ == "__main__":
    main()
