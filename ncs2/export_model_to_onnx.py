import os
import sys
sys.path.insert(1, os.path.join("..", "MSMatch"))
import torch
import argparse
from datasets.ssl_dataset import SSL_Dataset
from termcolor import colored
from utils import net_builder


def main():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, default="efficientnet-lite0")
    parser.add_argument("--load_path", type=str, default=r"C:\Users\meoni\Documents\ESA\Projects\END2END\MSMatch\checkpoint\thraws_swir_train\FixMatch_archefficientnet-lite0_batch8_confidence0.95_lr0.03_uratio4_wd0.00075_wu1.0_seed0_numlabels800_optSGD\model_best.pth")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--test_dataset", type=str, default="thraws_swir_test")
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen_factor", type=int, default=2)
    parser.add_argument("--leaky_slope", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    #Create SSL loaders
    data_dir=os.path.join(os.getcwd(),"..", "MSMatch", "DATA")
    print("Import "+colored("test", "green")+" dataset: "+colored(args.test_dataset,"blue"))
    _test_dset = SSL_Dataset(name=args.test_dataset, data_dir=data_dir)
    _ = _test_dset.get_dset()

    OUTPUT_DIR = args.output_dir

    int8_onnx_path=os.path.join(OUTPUT_DIR, args.model+".onnx")

    #Loading data
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

    net_eval = _net_builder(num_classes=_test_dset.num_classes, in_channels=_test_dset.num_channels)
    net_eval.load_state_dict(load_model_eval)

    net_eval.eval()


    # Create dummy input 
    dummy_input=torch.randn(1, _test_dset.num_channels, _test_dset.size[0], _test_dset.size[1])

    # Call the export function
    torch.onnx.export(net_eval, (dummy_input, ), int8_onnx_path)     

if __name__ == "__main__":
    main()