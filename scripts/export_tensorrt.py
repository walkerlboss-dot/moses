#!/usr/bin/env python3
"""
export_tensorrt.py

TensorRT optimization script for the Moses humanoid policy.

Loads an ONNX model, builds a TensorRT engine with FP16/INT8 support,
benchmarks inference speed, and saves the serialized .trt engine file.

Target: TensorRT 8.6+, CUDA 12.x, PyTorch 2.x
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Optional TensorRT import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TRT_AVAILABLE = True
except ImportError as _e:
    TRT_AVAILABLE = False
    TRT_IMPORT_ERROR = _e

# Optional ONNX Runtime for verification
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("export_tensorrt")


# ---------------------------------------------------------------------------
# TensorRT helpers
# ---------------------------------------------------------------------------
class TensorRTLogger:
    """Custom TensorRT logger that bridges to Python logging.
    Inherits from trt.ILogger when TensorRT is available."""

    def __init__(self, level: Any = None) -> None:
        if TRT_AVAILABLE:
            if level is None:
                level = trt.LoggerSeverity.INFO
            trt.ILogger.__init__(self)
        self.level = level

    def log(self, severity: Any, msg: str) -> None:
        if not TRT_AVAILABLE:
            logger.info("[TRT] %s", msg)
            return
        if severity <= self.level:
            if severity == trt.LoggerSeverity.VERBOSE:
                logger.debug("[TRT] %s", msg)
            elif severity == trt.LoggerSeverity.INFO:
                logger.info("[TRT] %s", msg)
            elif severity == trt.LoggerSeverity.WARNING:
                logger.warning("[TRT] %s", msg)
            elif severity == trt.LoggerSeverity.ERROR:
                logger.error("[TRT] %s", msg)
            elif severity == trt.LoggerSeverity.INTERNAL_ERROR:
                logger.critical("[TRT] %s", msg)


def build_engine(
    onnx_path: str,
    output_path: str,
    max_batch_size: int = 1,
    fp16: bool = True,
    int8: bool = False,
    workspace_mb: int = 4096,
    min_shapes: dict[str, list[int]] | None = None,
    opt_shapes: dict[str, list[int]] | None = None,
    max_shapes: dict[str, list[int]] | None = None,
) -> bool:
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to the ONNX model file.
        output_path: Path to save the serialized .trt engine.
        max_batch_size: Maximum batch size for the engine.
        fp16: Enable FP16 precision.
        int8: Enable INT8 precision (requires calibration).
        workspace_mb: Workspace size in MiB.
        min_shapes: Minimum input shapes for dynamic axes.
        opt_shapes: Optimal input shapes for dynamic axes.
        max_shapes: Maximum input shapes for dynamic axes.

    Returns:
        True if engine built successfully, False otherwise.
    """
    if not TRT_AVAILABLE:
        logger.error("TensorRT not available: %s", TRT_IMPORT_ERROR)
        return False

    logger.info("Building TensorRT engine from: %s", onnx_path)
    logger.info("  FP16: %s | INT8: %s | Workspace: %d MiB", fp16, int8, workspace_mb)

    logger_trt = TensorRTLogger(trt.LoggerSeverity.INFO)
    builder = trt.Builder(logger_trt)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger_trt)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()
    if not parser.parse(onnx_data):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        return False

    logger.info("ONNX parsed: %d inputs, %d outputs", network.num_inputs, network.num_outputs)

    # Config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")
    elif fp16:
        logger.warning("FP16 not supported on this platform")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("INT8 enabled (no calibration data provided)")
    elif int8:
        logger.warning("INT8 not supported on this platform")

    # Profile for dynamic shapes
    if min_shapes or opt_shapes or max_shapes:
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            name = input_tensor.name
            shape = list(input_tensor.shape)

            # Replace dynamic dims (-1) with provided shapes
            def resolve(s: list[int], shapes_dict: dict[str, list[int]] | None) -> list[int]:
                if shapes_dict and name in shapes_dict:
                    return shapes_dict[name]
                return [max(1, dim) if dim == -1 else dim for dim in s]

            min_s = resolve(shape, min_shapes)
            opt_s = resolve(shape, opt_shapes)
            max_s = resolve(shape, max_shapes)
            profile.set_shape(name, min_s, opt_s, max_s)
            logger.info("  Dynamic shape '%s': min=%s opt=%s max=%s", name, min_s, opt_s, max_s)

        config.add_optimization_profile(profile)

    # Build
    logger.info("Building engine (this may take a while)...")
    start = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    elapsed = time.time() - start

    if engine_bytes is None:
        logger.error("Engine build failed")
        return False

    # Save
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "wb") as f:
        f.write(engine_bytes)

    logger.info("Engine built in %.2fs | Size: %.2f MiB", elapsed, len(engine_bytes) / (1 << 20))
    logger.info("Saved: %s", output_path)
    return True


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
class TensorRTEngine:
    """Wrapper for a deserialized TensorRT engine."""

    def __init__(self, engine_path: str) -> None:
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.logger = TensorRTLogger(trt.LoggerSeverity.WARNING)
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Inspect bindings
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.bindings: list[Any] = []
        self.host_buffers: list[np.ndarray] = []
        self.device_buffers: list[Any] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # Allocate
            size = int(np.prod(shape)) if all(s > 0 for s in shape) else 0
            host_mem = cuda.pagelocked_empty(size, dtype) if size > 0 else np.array([], dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes) if size > 0 else 0

            self.bindings.append(int(device_mem) if size > 0 else 0)
            self.host_buffers.append(host_mem)
            self.device_buffers.append(device_mem)

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

            logger.debug("  Binding '%s': %s %s", name, shape, dtype)

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference."""
        # Copy inputs
        for name, arr in inputs.items():
            idx = self.engine.get_tensor_name.index(name) if hasattr(self.engine.get_tensor_name, "index") else None
            # Find index by iteration
            for i in range(self.engine.num_io_tensors):
                if self.engine.get_tensor_name(i) == name:
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"Input '{name}' not found")

            # Set shape if dynamic
            self.context.set_input_shape(name, arr.shape)
            self.host_buffers[idx][: arr.size] = arr.ravel()
            cuda.memcpy_htod_async(self.device_buffers[idx], self.host_buffers[idx], self.stream)

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs
        outputs: dict[str, np.ndarray] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.OUTPUT:
                shape = self.context.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                cuda.memcpy_dtoh_async(self.host_buffers[i], self.device_buffers[i], self.stream)
                self.stream.synchronize()
                out_arr = np.copy(self.host_buffers[i][: int(np.prod(shape))])
                outputs[name] = out_arr.reshape(shape)

        return outputs

    def benchmark(
        self,
        input_shapes: dict[str, tuple[int, ...]],
        warmup: int = 10,
        iterations: int = 100,
    ) -> dict[str, float]:
        """Benchmark inference latency and throughput."""
        # Create dummy inputs
        inputs = {}
        for name, shape in input_shapes.items():
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            inputs[name] = np.random.randn(*shape).astype(dtype)

        # Warmup
        for _ in range(warmup):
            self.infer(inputs)

        # Benchmark
        latencies: list[float] = []
        for _ in range(iterations):
            start = time.time()
            self.infer(inputs)
            latencies.append((time.time() - start) * 1000.0)  # ms

        latencies_arr = np.array(latencies)
        batch_size = next(iter(input_shapes.values()))[0]

        return {
            "mean_latency_ms": float(latencies_arr.mean()),
            "std_latency_ms": float(latencies_arr.std()),
            "min_latency_ms": float(latencies_arr.min()),
            "max_latency_ms": float(latencies_arr.max()),
            "p50_latency_ms": float(np.percentile(latencies_arr, 50)),
            "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
            "p99_latency_ms": float(np.percentile(latencies_arr, 99)),
            "throughput_hz": 1000.0 / float(latencies_arr.mean()),
            "batch_size": batch_size,
            "iterations": iterations,
        }


# ---------------------------------------------------------------------------
# ONNX Runtime benchmark (fallback / comparison)
# ---------------------------------------------------------------------------
def benchmark_onnx(onnx_path: str, input_shapes: dict[str, tuple[int, ...]], iterations: int = 100) -> dict[str, float] | None:
    """Benchmark ONNX Runtime inference."""
    if not ORT_AVAILABLE:
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    inputs = {}
    for name, shape in input_shapes.items():
        inputs[name] = np.random.randn(*shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, inputs)

    latencies: list[float] = []
    for _ in range(iterations):
        start = time.time()
        session.run(None, inputs)
        latencies.append((time.time() - start) * 1000.0)

    latencies_arr = np.array(latencies)
    return {
        "mean_latency_ms": float(latencies_arr.mean()),
        "std_latency_ms": float(latencies_arr.std()),
        "throughput_hz": 1000.0 / float(latencies_arr.mean()),
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize Moses policy with TensorRT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onnx", type=str, required=True, help="Path to ONNX model"
    )
    parser.add_argument(
        "--output", type=str, default="", help="Output .trt path (default: auto)"
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=1, help="Max batch size"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Enable FP16"
    )
    parser.add_argument(
        "--int8", action="store_true", help="Enable INT8"
    )
    parser.add_argument(
        "--workspace-mb", type=int, default=4096, help="Workspace size (MiB)"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run inference benchmark"
    )
    parser.add_argument(
        "--benchmark-iterations", type=int, default=1000, help="Benchmark iterations"
    )
    parser.add_argument(
        "--input-shape", type=str, default="obs:1x69",
        help="Input shape spec, e.g. 'obs:1x69' or 'obs:1,69'"
    )
    parser.add_argument(
        "--compare-onnx", action="store_true", help="Compare against ONNX Runtime"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./trt_engines", help="Output directory"
    )
    return parser.parse_args()


def parse_shape_spec(spec: str) -> dict[str, tuple[int, ...]]:
    """Parse 'name:1x69' or 'name:1,69' into dict."""
    result: dict[str, tuple[int, ...]] = {}
    for part in spec.split(";"):
        if ":" not in part:
            continue
        name, dims_str = part.split(":", 1)
        dims = tuple(int(x) for x in dims_str.replace("x", ",").split(","))
        result[name.strip()] = dims
    return result


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    logger.info("=" * 60)
    logger.info("Moses Policy TensorRT Optimization")
    logger.info("=" * 60)

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        logger.error("ONNX file not found: %s", onnx_path)
        return 1

    # Auto output path
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "_fp16" if args.fp16 else "_fp32"
        if args.int8:
            suffix += "_int8"
        output_path = Path(args.output_dir) / (onnx_path.stem + suffix + ".trt")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse input shapes
    input_shapes = parse_shape_spec(args.input_shape)
    logger.info("Input shapes: %s", input_shapes)

    # Build engine
    success = build_engine(
        onnx_path=str(onnx_path),
        output_path=str(output_path),
        max_batch_size=args.max_batch_size,
        fp16=args.fp16,
        int8=args.int8,
        workspace_mb=args.workspace_mb,
        opt_shapes=input_shapes,
        max_shapes={k: tuple(max(d, 64) for d in v) for k, v in input_shapes.items()},
    )

    if not success:
        logger.error("Engine build failed")
        return 1

    # Benchmark
    results: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "engine_path": str(output_path),
        "fp16": args.fp16,
        "int8": args.int8,
        "max_batch_size": args.max_batch_size,
    }

    if args.benchmark and TRT_AVAILABLE:
        logger.info("Running TensorRT benchmark...")
        try:
            engine = TensorRTEngine(str(output_path))
            trt_results = engine.benchmark(
                input_shapes=input_shapes,
                warmup=50,
                iterations=args.benchmark_iterations,
            )
            results["tensorrt"] = trt_results
            logger.info("TensorRT latency: %.3f ± %.3f ms | %.1f Hz",
                        trt_results["mean_latency_ms"],
                        trt_results["std_latency_ms"],
                        trt_results["throughput_hz"])
        except Exception as exc:
            logger.exception("TensorRT benchmark failed: %s", exc)

    if args.compare_onnx:
        logger.info("Running ONNX Runtime benchmark...")
        onnx_results = benchmark_onnx(str(onnx_path), input_shapes, iterations=args.benchmark_iterations)
        if onnx_results:
            results["onnxruntime"] = onnx_results
            logger.info("ONNX Runtime latency: %.3f ± %.3f ms | %.1f Hz",
                        onnx_results["mean_latency_ms"],
                        onnx_results["std_latency_ms"],
                        onnx_results["throughput_hz"])

    # Save results
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved: %s", json_path)

    logger.info("Optimization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
