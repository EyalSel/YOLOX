#!/usr/bin/env python3
"""
YOLOX Latency Profiler

This script profiles the inference latency of YOLOX models with
different configurations. It measures the time taken for inference across
multiple iterations and reports p10, p50, and p90 latency statistics in
milliseconds.

The script uses the YOLOX inference utilities to load models and perform
inference. It first performs warmup iterations to ensure the model is fully
loaded and optimized, then measures latency over a specified number of
iterations.

Results are saved to a JSON file with a naming convention that includes the
model name, batch size, and resolution.

Usage:
    python profile_yolox_latency.py --model=yolox-tiny --batchsize=4
    --width=512 --height=512

Required flags:
    --model: YOLOX model to profile (e.g., yolox-tiny, yolox-s, yolox-m, etc.)
    --batchsize: Batch size for inference
    --width: Input width
    --height: Input height

Optional flags:
    --output_dir: Directory to save the latency profile (default:
    ./latency_profiles)
    --warmup_iters: Number of warmup iterations (default: 100)
    --measure_iters: Number of measurement iterations (default: 200)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from absl import app, flags
from loguru import logger
from tqdm import tqdm

from yolox.data.data_augment import ValTransform
from yolox.exp.build import get_exp_by_name
from yolox.utils import get_model_info, postprocess

FLAGS = flags.FLAGS

ckpt_paths = {
    "yolox-tiny": "model_ckpts/yolox_tiny.pth",
    "yolox-m": "model_ckpts/yolox_m.pth",
    "yolox-l": "model_ckpts/yolox_l.pth",
    "yolox-s": "model_ckpts/yolox_s.pth",
    "yolox-nano": "model_ckpts/yolox_nano.pth",
    "yolox-x": "model_ckpts/yolox_x.pth",
}

flags.DEFINE_enum("model", None, list(ckpt_paths.keys()),
                  "YOLOX model to profile")
flags.DEFINE_integer("batchsize", None, "Batch size for inference")
flags.DEFINE_integer("width", None, "Input width")
flags.DEFINE_integer("height", None, "Input height")
flags.DEFINE_string("output_dir", "./latency_profiles",
                    "Directory to save the latency profile")
flags.DEFINE_integer("warmup_iters", 100, "Number of warmup iterations")
flags.DEFINE_integer("measure_iters", 200, "Number of measurement iterations")

flags.mark_flag_as_required("model")
flags.mark_flag_as_required("batchsize")
flags.mark_flag_as_required("width")
flags.mark_flag_as_required("height")


def preprocess(images, test_size):
    """
    Preprocess images for YOLOX inference.
    
    Args:
        images: Batch of images in [N, H, W, C] format, uint8 (0-255)
        test_size: Tuple of (height, width) for model input
        
    Returns:
        Preprocessed tensor ready for model input and the resize ratio
    """
    # Validate input images
    assert isinstance(images, np.ndarray), "Images must be a numpy array"
    assert images.ndim == 4, (
        f"Images must be 4D array [N, H, W, C], got shape {images.shape}")
    assert images.shape[0] > 0, "Batch size must be greater than 0"
    assert images.shape[
        3] == 3, f"Images must have 3 channels, got {images.shape[3]}"
    assert images.dtype == np.uint8, (
        f"Images must be uint8, got {images.dtype}")
    # Preprocess images
    preproc = ValTransform(legacy=False)
    height, width = images.shape[1], images.shape[2]
    ratio = min(test_size[0] / height, test_size[1] / width)

    pre_proc_images = np.array(
        [preproc(img, None, test_size)[0] for img in images])
    img = torch.from_numpy(pre_proc_images)
    img = img.float()

    return img, ratio


def postprocess_outputs_cpu(outputs, ratio, batch_size):
    """
    Postprocess YOLOX model outputs into a standardized detection format.
    
    Args:
        outputs: Raw outputs from YOLOX model
        num_classes: Number of classes for the model
        confthre: Confidence threshold
        nmsthre: NMS threshold
        ratio: Resize ratio used in preprocessing
        batch_size: Number of images in the batch
        
    Returns:
        Processed detection results in standardized format
    """
    # Validate outputs format and on cpu
    assert isinstance(outputs, list), "Outputs must be a list"
    for output in outputs:
        if output is None:
            continue
        assert isinstance(
            output, np.ndarray), "Each output must be None or a numpy array"
        assert output.device.type == "cpu", "Each output must be on CPU"

    # pad outputs to 100 with zeros
    outputs = np.array([
        np.concatenate([x, np.zeros([100 - x.shape[0], x.shape[1]])], axis=0)
        if x.shape[0] < 100 else x[:100] for x in outputs
    ])

    if outputs.size == 0 or outputs.shape[1] == 0:
        # Return empty results with correct shape
        return np.zeros((batch_size, 100, 7))

    # reorganize columns
    class_label = outputs[:, :, 6]
    scores = outputs[:, :, 4] * outputs[:, :, 5]
    xmin = outputs[:, :, 0] / ratio
    ymin = outputs[:, :, 1] / ratio
    xmax = outputs[:, :, 2] / ratio
    ymax = outputs[:, :, 3] / ratio

    result = np.stack(
        [-np.ones_like(ymin), ymin, xmin, ymax, xmax, scores, class_label],
        axis=0).transpose([1, 2, 0])
    return result


def gpu_inference(model, img, num_classes, confthre, nmsthre):
    """
    Measures the end-to-end GPU inference time of a YOLOX model.
    """
    img = img.cuda()
    with torch.no_grad():
        outputs = model(img)
    outputs = postprocess(outputs,
                          num_classes,
                          confthre,
                          nmsthre,
                          class_agnostic=True)
    outputs = [
        x.cpu().detach().numpy() if x is not None else np.zeros((0, 7))
        for x in outputs
    ]
    return outputs


# Currently not used, because preprocessing should be outside of the critical
# path of latency measurement
def run_inference(model, images, test_size, num_classes, confthre, nmsthre):
    """
    Run inference on a batch of images with the YOLOX model.
    
    Args:
        model: YOLOX model
        images: Batch of images in [N, H, W, C] format, uint8 (0-255)
        test_size: Tuple of (height, width) for model input
        num_classes: Number of classes for the model
        confthre: Confidence threshold
        nmsthre: NMS threshold
        
    Returns:
        Processed detection results
    """
    # Preprocessing
    img, ratio = preprocess(images, test_size)

    # Model inference
    outputs = gpu_inference(model, img, num_classes, confthre, nmsthre)

    # Postprocessing
    results = postprocess_outputs_cpu(outputs, num_classes, confthre, nmsthre,
                                      ratio, images.shape[0])

    return results


def setup_model(model_name, conf=0.05, nms=0.45, tsize=640):
    """Sets up and returns a YOLOX model for inference.
    
    Args:
        model_name: Name of YOLOX model variant (e.g. "yolox-tiny", "yolox-s",
        etc)
        conf: Confidence threshold for detections
        nms: NMS threshold
        tsize: Input image size
    
    Returns:
        model: Loaded YOLOX model ready for inference
        exp: YOLOX experiment configuration
    """
    exp = get_exp_by_name(model_name)
    exp.test_conf = conf
    exp.nmsthre = nms
    exp.test_size = (tsize, tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(
        model, exp.test_size)))

    model.cuda()
    model.eval()

    ckpt_path = ckpt_paths[model_name]

    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    return model, exp


def profile_model_latency(model_name,
                          batchsize,
                          width,
                          height,
                          warmup_iters=100,
                          measure_iters=200):
    """
    Profile the latency of a YOLOX model with specified parameters.
    
    Args:
        model_name: YOLOX model name (e.g., 'yolox-tiny')
        batchsize: Batch size for inference
        width: Input width
        height: Input height
        warmup_iters: Number of warmup iterations
        measure_iters: Number of measurement iterations
    
    Returns:
        Dictionary with p10, p50, and p90 latency statistics in milliseconds
    """
    print(
        f"Profiling {model_name} with batchsize={batchsize}, width={width}, height={height}"
    )

    # Use the maximum of width and height for the model's resolution parameter
    model_resolution = max(width, height)
    print(f"Using model resolution: {model_resolution}")

    # Setup model
    conf = 0.05
    nms = 0.45
    tsize = model_resolution
    model, exp = setup_model(model_name, conf, nms, tsize)

    # Pre-generate random images for warmup
    print("Generating random image batches for warmup...")
    warmup_images = np.random.randint(
        0, 256, (warmup_iters, batchsize, height, width, 3), dtype=np.uint8)

    # Pre-generate random images for measurement
    print("Generating random image batches for measurement...")
    measurement_images = np.random.randint(
        0, 256, (measure_iters, batchsize, height, width, 3), dtype=np.uint8)

    # Pre-process all images before measurement to remove preprocessing from critical path
    print("Pre-processing all images...")
    preprocessed_warmup = []
    for i in tqdm(range(warmup_iters), desc="Pre-processing warmup images"):
        img, ratio = preprocess(warmup_images[i], (tsize, tsize))
        preprocessed_warmup.append((img, ratio))

    preprocessed_measurement = []
    for i in tqdm(range(measure_iters),
                  desc="Pre-processing measurement images"):
        img, ratio = preprocess(measurement_images[i], (tsize, tsize))
        preprocessed_measurement.append((img, ratio))

    # Warmup
    print(f"Running {warmup_iters} warmup iterations...")
    for i in tqdm(range(warmup_iters), desc="Warmup"):
        img, ratio = preprocessed_warmup[i]
        outputs = gpu_inference(model, img, exp.num_classes, conf, nms)
        # _ = postprocess_outputs_cpu(outputs, exp.num_classes, conf, nms, ratio, batchsize)

    # Measurement - only measure GPU inference time
    print(f"Running {measure_iters} measurement iterations...")
    latencies = []
    for i in tqdm(range(measure_iters), desc="Measurement"):
        img, ratio = preprocessed_measurement[i]

        # Synchronize CUDA before timing
        torch.cuda.synchronize()
        start_time = time.time()

        outputs = gpu_inference(model, img, exp.num_classes, conf, nms)

        # Synchronize CUDA to ensure all GPU operations are complete
        torch.cuda.synchronize()
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency_ms)

        # Postprocessing (outside of timing)
        # _ = postprocess_outputs_cpu(outputs, exp.num_classes, conf, nms, ratio, batchsize)

    # Calculate statistics
    p10 = np.percentile(latencies, 10)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)

    return {
        "model": model_name,
        "batchsize": batchsize,
        "width": width,
        "height": height,
        "model_resolution": model_resolution,
        "latency_ms": {
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90)
        },
        "num_iterations": measure_iters
    }


def main(_):
    # Create output directory if it doesn't exist
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Profile the model
    results = profile_model_latency(model_name=FLAGS.model,
                                    batchsize=FLAGS.batchsize,
                                    width=FLAGS.width,
                                    height=FLAGS.height,
                                    warmup_iters=FLAGS.warmup_iters,
                                    measure_iters=FLAGS.measure_iters)

    # Save results to a JSON file
    output_file = output_dir / (
        "model_latency--model={}__batchsize={}__resolution={}x{}.json".format(
            FLAGS.model, FLAGS.batchsize, FLAGS.width, FLAGS.height))
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")
    print(f"Latency statistics (ms): p10={results['latency_ms']['p10']:.2f}, "
          f"p50={results['latency_ms']['p50']:.2f}, "
          f"p90={results['latency_ms']['p90']:.2f}")


if __name__ == "__main__":
    app.run(main)
