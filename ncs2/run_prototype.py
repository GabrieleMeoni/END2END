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
from pathlib import Path
from onboard_detector_processor import oboardDetectorProcessor
from onboard_prototype_utils import load_data, plot
import numpy as np

def plot_results(x_coreg_list, y_list, save_fig_path):
    fig, ax=plt.subplots(3,2, figsize=(60,60))

    for n in range(3):
        x=x_coreg_list[n]
        y=y_list[n]
        plot(x,x_max=1, ax=ax[n,0])
        ax[n,0].set_title("Image")
        ax[n,1].set_title("Predicted mask")
        #x_map=torch.zeros([x.shape[1], x.shape[2], 3])
        ax[n,1].set_xlim(0, x.shape[2])
        ax[n,1].set_ylim(0, x.shape[1])
        patch_list=[]
        for h in range(y.shape[1]):
            for v in range(y.shape[0]):
                if y[v,h] == 0:
                    patch_list.append(Rectangle((h * 256,x.shape[1] - v * 256), 256, -256, color='red'))
        for patch in patch_list:
            ax[n,1].add_patch(patch)
    plt.savefig(save_fig_path)

def main():
    
    parser = argparse.ArgumentParser(description="")

    ### DETECTOR AND SATELLITE ARGUMENTS
    parser.add_argument("--event_name", type=str)
    parser.add_argument("--detector_number", type=int)
    parser.add_argument("--dataset_path", type=str, default=r"C:\Users\meoni\Documents\ESA\Projects\END2END\onboard_prototype\my_dir")
    
    ### ONNX ARGUMENTS
    parser.add_argument("--ir_path", type=str, default=r"C:\Users\meoni\Documents\ESA\Projects\END2END\ncs2\output\efficientnet-lite0.xml")

    ### OUTPUT DIRECTORY ARGUMENTS
    parser.add_argument("--output_dir", type=str, default="output")

    ### EMBEDDED DEVICE ARGUMENTS
    parser.add_argument("--device", type=str, default="MYRIAD", help="Embedded device. Supported ""CPU"", ""GPU"", ""MYRIAD"".")

    args = parser.parse_args()

    ## Loading data
    data_loaded=load_data(args.dataset_path)

    detector_number_str=str(args.detector_number)

    #Selecting the event of interests
    try:
        event=data_loaded[args.event_name]
    except:
        raise ValueError(f"Impossible to find the event: "+colored(args.event_name, "red")+".")
    
    if not(detector_number_str in list(event[1].keys())):
        raise ValueError(f"The event: "+colored(args.event_name, "red")+" has no granule with detector number: "+colored(detector_number_str)+".")

    # Granules numbers and lists
    granules_numbers_list=[list(event[1][detector_number_str].keys())[::-1] if not(int(args.detector_number) % 2) else list(event[1][detector_number_str].keys())][0]
    granules_dict=event[1][detector_number_str]


    # Load onboard processor detector 
    onboard_processor=oboardDetectorProcessor(event[0], args.detector_number, model_name=None)

    # IR path
    ir_path=args.ir_path

    print("Compiling model: "+colored(ir_path,"blue"))
    ie_core = ov.Core()
    # Read the network and corresponding weights from a file.
    model = ie_core.read_model(model=ir_path)
    # Compile the model for CPU (you can choose manually CPU, GPU, MYRIAD etc.)
    # or let the engine choose the best available device (AUTO).
    compiled_model = ie_core.compile_model(model=model, device_name=args.device)

    # Get the input and output nodes.
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print(colored("Start testing...", "red"))
    
    # Results
    start_time = time.time()
    masks_list=[]
    x_coreg_list=[]
    for granule_number in granules_numbers_list:
        granule_coreg=onboard_processor.coarse_coregistration(granules_dict[granule_number]/255)
        onboard_processor.init_patch_engine(granule_coreg)
        # Size of the output mask
        n_v_max, n_h_max=onboard_processor.get_output_shape()

        y_demosaicked=np.zeros([n_v_max, n_h_max])
        n_v,n_h=0,0

        while not(onboard_processor.processing_finished()):
            granule_patch=onboard_processor.get_next_patch()
            y_demosaicked[n_v, n_h]= compiled_model([granule_patch.unsqueeze(0)])[output_layer].argmax()
            if n_v < n_v_max - 1:
                n_v +=1
            else: 
                n_v = 0
                n_h +=1
    
        
        masks_list.append(y_demosaicked)
        x_coreg_list.append(granule_coreg)
    
    stop_time = time.time()
    N_processed=len(granules_numbers_list)
    inference_time_s = (stop_time - start_time)

    print(colored("Testing finished.", "green"))
    print(colored("Calculating results...", "blue"))

    print("Total inference time[s]: "+colored(str(inference_time_s), "blue"))
    print("Number of granules: "+colored(str(N_processed), "yellow"))
    print("Average inference time per granule[s]: "+colored(str(inference_time_s/N_processed), "cyan"))

    plot_results(x_coreg_list, masks_list, args.event_name+"_"+str(args.detector_number)+"_mask.png")


if __name__ == "__main__":
    main()