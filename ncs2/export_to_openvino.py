import os
import sys
sys.path.insert(1, os.path.join("..", "..", "MSMatch"))
import argparse
from pathlib import Path

def main():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--onnx_path", type=str, default=r"C:\Users\meoni\Documents\ESA\Projects\END2END\ncs2\openvino\output\efficientnet-lite0.onnx")
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()


    ir_path = Path(args.onnx_path).with_suffix(".xml")
    # Construct the command for Model Optimizer.
    mo_command = f"""mo
                    --input_model "{args.onnx_path}"
                    --compress_to_fp16
                    --output_dir "{ir_path.parent}"
                    """
    mo_command = " ".join(mo_command.split())
    print(mo_command)

    print("Model Optimizer command to convert the ONNX model to OpenVINO:")

    if not ir_path.exists():
        print("Exporting ONNX model to IR... This may take a few minutes.")
        mo_result = os.system(mo_command)
    else:
        print(f"IR model {ir_path} already exists.")

if __name__ == "__main__":
    main()