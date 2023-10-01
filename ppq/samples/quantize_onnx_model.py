from typing import Iterable

import torch
from torch.utils.data import DataLoader

from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_onnx_model

BATCHSIZE = 32
INPUT_SHAPE = [3, 224, 224]
DEVICE = "cuda"  # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.TRT_INT8  # identify a target platform for your network.
ONNX_PATH = "Models/cls_model/mobilenet_v2.onnx"


def load_calibration_dataset() -> Iterable:
    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)


quant_setting = QuantizationSettingFactory.trt_setting()

# Load training data for creating a calibration dataloader.
calibration_dataset = load_calibration_dataset()
calibration_dataloader = DataLoader(
    dataset=calibration_dataset, batch_size=BATCHSIZE, shuffle=True
)

# quantize your model.
quantized = quantize_onnx_model(
    onnx_import_file=ONNX_PATH,
    calib_dataloader=calibration_dataloader,
    calib_steps=32,
    input_shape=[BATCHSIZE] + INPUT_SHAPE,
    setting=quant_setting,
    collate_fn=collate_fn,
    platform=PLATFORM,
    device=DEVICE,
    verbose=0,
)

# Quantization Result is a PPQ BaseGraph instance.
assert isinstance(quantized, BaseGraph)

# export quantized graph.
export_ppq_graph(
    graph=quantized,
    platform=PLATFORM,
    graph_save_to="Output/quantized(onnx).onnx",
    config_save_to="Output/quantized(onnx).json",
)
