## Inspired by: https://docs.openvino.ai/latest/notebooks/302-pytorch-quantization-aware-training-with-output.html
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.insert(1, os.path.join("..", "..", "MSMatch"))
import warnings  # To disable warnings on export to ONNX.

from pathlib import Path
import logging

import torch
import nncf  # Important - should be imported directly after torch.


import torch.nn.parallel
import torch.nn as nn
import torch.optim

from nncf.common.logging.logger import set_log_level
set_log_level(logging.ERROR)  # Disables all NNCF info and warning messages.
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from openvino.runtime import Core
from torch.jit import TracerWarning
import argparse
from datasets.ssl_dataset import SSL_Dataset
from termcolor import colored
from utils import net_builder
from quantize_utils import train, validate
import warnings
warnings.filterwarnings("ignore",category=UserWarning)



def export_complier_to_LIB(cl_exe_path):
    if sys.platform == "win32":
        import distutils.command.build_ext
        import os

        os.environ["PATH"] += f"{os.pathsep}{cl_exe_path}"
        d = distutils.core.Distribution()
        b = distutils.command.build_ext.build_ext(d)
        b.finalize_options()
        os.environ["LIB"] = os.pathsep.join(b.library_dirs)
    return

def main():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed",type=int, default=0, help="Seed to use.")
    parser.add_argument("--model", type=str, default="efficientnet-lite0")
    parser.add_argument("--cl_exe_path", type=str, default=r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.35.32215\bin\Hostx64\x64")
    parser.add_argument("--load_path", type=str, default=r"C:\Users\meoni\Documents\ESA\Projects\END2END\MSMatch\checkpoint\thraws_swir_train\FixMatch_archefficientnet-lite0_batch8_confidence0.95_lr0.03_uratio4_wd0.00075_wu1.0_seed0_numlabels800_optSGD\model_best.pth")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--train_dataset", type=str, default="thraws_swir_train")
    parser.add_argument("--test_dataset", type=str, default="thraws_swir_test")
    parser.add_argument("--train_upsample_event", type=int, default=1)
    parser.add_argument("--train_upsample_notevent", type=int, default=1)
    parser.add_argument("--eval_upsample_event", type=int, default=1)
    parser.add_argument("--eval_upsample_notevent", type=int, default=1)
    parser.add_argument("--eval_split_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs_ft", type=int, default=4, help="Epochs for fine tuning.")
    parser.add_argument("--lr_init", type=float, default=0.03, help="Initial learning rate.")
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--leaky_slope", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    export_complier_to_LIB(args.cl_exe_path)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    #Create SSL loaders
    data_dir=os.path.join(os.getcwd(), "..","..", "MSMatch", "DATA")
    _train_dset = SSL_Dataset(name=args.train_dataset, train=True,  data_dir=data_dir, seed=args.seed, eval_split_ratio=args.eval_split_ratio,upsample_event=args.train_upsample_event,upsample_notevent=args.train_upsample_notevent)
    _test_dset = SSL_Dataset(name=args.test_dataset, data_dir=data_dir)
    
    
    OUTPUT_DIR = args.output_dir

    #Loading data
    print("Import "+colored("train", "green")+" dataset: "+colored(args.train_dataset,"blue"))
    train_dset = _train_dset.get_dset()
    print("Import "+colored("test", "green")+" dataset: "+colored(args.test_dataset,"blue"))
    test_dset = _test_dset.get_dset()
    os.makedirs(OUTPUT_DIR, exist_ok=True)




    # Read pretrained model
    checkpoint_path = os.path.join(args.load_path)
    print("Load model: "+colored(args.model, "green")+" from checkpoint: "+colored(checkpoint_path,"blue"))
    
    checkpoint = torch.load(checkpoint_path)
    load_model_eval = (checkpoint["eval_model"])

    _net_builder = net_builder(
        args.model,
        False,
        {
            "depth": args.depth,
            "widen_factor": args.widen_factor,
            "leaky_slope": args.leaky_slope,
            "dropRate": args.dropout,
        },
    )

    net_eval = _net_builder(num_classes=_train_dset.num_classes, in_channels=_train_dset.num_channels)
    net_eval.load_state_dict(load_model_eval)

    if torch.cuda.is_available():
        net_eval=net_eval.cuda()

    net_eval.eval()


    # Get dataloaders 
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, sampler=None
    )

    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    # Test float model
    print(colored("FLOAT32 model", "blue")+" testing...")
    acc_float32, mcc_float32= validate(test_loader, net_eval, device)
    print(colored("FLOAT32 model", "blue")+":\n\t\tTest accuracy: "+colored(str(acc_float32), "green") +"\n\t\tTest mcc: "+colored(str(mcc_float32), "green")+"\n\n")

    # Quantization parameters. 
    nncf_config_dict = {
    "input_info": {"sample_size": [1, _train_dset.num_channels, _train_dset.size[0], _train_dset.size[1]]},
    "log_dir": str(OUTPUT_DIR),  # The log directory for NNCF-specific logging outputs.
    "compression": {
        "algorithm": "quantization",  # Specify the algorithm here.
        },
    }
    # Configuring quatization parameters.
    nncf_config = NNCFConfig.from_dict(nncf_config_dict)

    # Specifying dataloader to optimize quatization.
    nncf_config = register_default_init_args(nncf_config, train_loader)

    # Compress model - INT8 
    ## Forcing use of CPU for incompatibility of NCC with quantization
    net_eval.to(torch.device("cpu"))
    print(colored("Model quantization ongoing...", "red"))
    compression_ctrl, net_eval_int8 = create_compressed_model(net_eval, nncf_config)
    print("Model quantization "+colored("completed", "green")+".")

    # Test int8 model
    print(colored("INT8 model", "blue")+" testing...")
    acc_int8, mcc_int8= validate(test_loader, net_eval_int8, torch.device("cpu"))
    print(colored("INT8 model", "blue")+":\n\t\tTest accuracy: "+colored(str(acc_int8), "green") +"\n\t\tTest mcc: "+colored(str(mcc_int8), "green")+"\n\n")

    # Start fine-tuning.

    criterion = nn.CrossEntropyLoss().to(torch.device("cpu"))
    compression_lr = args.lr_init / 10
    optimizer = torch.optim.Adam(net_eval_int8.parameters(), lr=compression_lr)

    # Train for one epoch with NNCF.
    train(train_loader, net_eval_int8, criterion, optimizer, epoch=args.epochs_ft, device=torch.device("cpu"))
    

if __name__ == "__main__":
    main()