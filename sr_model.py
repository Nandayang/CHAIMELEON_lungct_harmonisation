import numpy as np
import torch
import argparse
from utils import image_processing_utils
from utils import io_utils
from utils import model_utils
import os

def predict(
    input_file_path: str,
    output_file_path: str,
    weight_file_path: str,
    use_gpu: int,
    lungmask_path: str):
    input, min, max, dtype = _pre_process_input(input_file_path, use_gpu, lungmask_path)
    model = model_utils.build_model(weight_file_path, use_gpu)

    prediction = model(input)
    prediction = _post_process_prediction(prediction, min, max, dtype)

    io_utils.save_nifti_image(
        prediction,
        target_file_path = output_file_path,
        reference_file_path = input_file_path,
    )

def _pre_process_input(
        input_file_path: str, use_gpu: int,lungmask_path:str):
    input, min, max, dtype = _pre_process_input_array(input_file_path, lungmask_path)
    input = _pre_process_input_tensor(input, use_gpu)

    return input, min, max, dtype

def _pre_process_input_array(input_file_path: str,lungmask_modelpath: str):
    segmentation = image_processing_utils.segment_lung(input_file_path, lungmask_modelpath)

    input = io_utils.load_nifti_image(input_file_path)
    input = image_processing_utils.extract_lung(input, segmentation)
    input, min, max, dtype = image_processing_utils.normalise_image(input)

    return input, min, max, dtype

def _pre_process_input_tensor(input: np.ndarray, use_gpu: bool,):
    input = image_processing_utils.convert_array_to_tensor(input)
    input = image_processing_utils.upsample_image(input)
    input = input.float()

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return input.to(device)

def _post_process_prediction(
    prediction: torch.Tensor, min: float, max: float, dtype: str,):
    prediction = image_processing_utils.convert_tensor_to_array(prediction)
    prediction = image_processing_utils.recover_image(
        prediction, min, max, dtype,
    )

    return prediction


parser = argparse.ArgumentParser('Slice generator from IC')
parser.add_argument('-i', "--inputs", type=str, help="input path of the Low-resolution CT scans")
parser.add_argument('-o', "--outputs", type=str, help="savepath of the harmonised results")
parser.add_argument('-g', "--gpu", type=int, default=-1, help="use gpu or cpu(-1)")
args = parser.parse_args()

if __name__ == "__main__":
    fids = os.listdir(args.inputs)
    for fid in fids:
        print("processing ", fid)
        predict(input_file_path=args.inputs+fid, output_file_path=args.outputs+fid, weight_file_path="weight.pth",use_gpu=args.gpu, lungmask_path="unet_r231-d5d2fc3d.pth")
