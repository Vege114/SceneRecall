#!/usr/bin/env python3
"""Measure real LR-ASD optimizer updates on synthetic, pinned CPU inputs.

Run with the LR-ASD environment under Linux/WSL; no source or checkpoint is changed.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback

SCRIPT_START = time.perf_counter()


def proc_values(path):
    values = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, value = line.partition(":")
            parts = value.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0])
    except OSError:
        pass
    return values


def memory_snapshot():
    process = proc_values("/proc/self/status")
    system = proc_values("/proc/meminfo")
    swap = {}
    try:
        swap = {key: int(value) for key, value in
                (line.split() for line in Path("/proc/vmstat").read_text().splitlines())
                if key in ("pswpin", "pswpout")}
    except OSError:
        pass
    return {
        "process_kib": {key: process.get(key) for key in
                        ("VmRSS", "VmHWM", "VmSize", "VmSwap")},
        "wsl_kib": {key: system.get(key) for key in
                    ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree")},
        "swap_pages": swap,
    }


def command_output(args, cwd=None):
    try:
        return subprocess.check_output(args, cwd=cwd, text=True,
                                       stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return "unavailable: " + str(error)


class GpuMonitor:
    fields = ("index", "name", "memory.used", "utilization.gpu", "power.draw",
              "temperature.gpu", "clocks.sm", "clocks.mem")

    def __init__(self):
        self.stop_event = threading.Event()
        self.samples = []
        self.errors = []
        self.phase = "setup"
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            started = time.perf_counter()
            phase = self.phase
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=" + ",".join(self.fields),
                     "--format=csv,noheader,nounits"], text=True,
                    stderr=subprocess.STDOUT, timeout=3)
                for row in csv.reader(output.splitlines()):
                    sample = {"elapsed_sec": started - SCRIPT_START, "phase": phase}
                    for key, value in zip(self.fields, row):
                        value = value.strip()
                        try:
                            sample[key] = float(value)
                        except ValueError:
                            sample[key] = value
                    self.samples.append(sample)
            except (OSError, subprocess.SubprocessError) as error:
                message = str(error)
                if message not in self.errors:
                    self.errors.append(message)
            self.stop_event.wait(max(0, 1 - (time.perf_counter() - started)))

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=4)


def summarize(values, np):
    if not values:
        return None
    return {"count": len(values), "total_sec": sum(values),
            "mean_sec": statistics.mean(values), "median_sec": statistics.median(values),
            "min_sec": min(values), "max_sec": max(values),
            "p10_sec": float(np.percentile(values, 10)),
            "p90_sec": float(np.percentile(values, 90)),
            "stdev_sec": statistics.stdev(values) if len(values) > 1 else 0.0}


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/d/SceneRecall-LR-ASD"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/mnt/d/SceneRecall/test-results/lrasd-200step-20261015"))
    parser.add_argument("--weights", type=Path, default=Path("weight/pretrain_AVA.model"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20,
                        help="Initial updates excluded from steady statistics, included in --steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    if min(args.batch_size, args.frames, args.steps) < 1 or not 0 <= args.warmup < args.steps:
        parser.error("Positive batch/frames/steps required; 0 <= warmup < steps")
    args.root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = args.weights if args.weights.is_absolute() else args.root / args.weights
    monitor = GpuMonitor()
    result = {
        "status": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {**{key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
                          "weights": str(weights), "precision": "FP32; TF32 disabled; no AMP",
                          "visual_shape": [args.batch_size, args.frames, 112, 112],
                          "audio_shape": [args.batch_size, args.frames * 4, 13],
                          "frames_per_update": args.batch_size * args.frames,
                          "optimizer": "upstream AdamW, all parameters, weight_decay=0.01",
                          "objective": "upstream lossAV + 0.5 * lossV",
                          "learning_rate_policy": "constant explicit fine-tuning LR; no scheduler step",
                          "input_source": "fixed seeded synthetic pinned CPU tensors; H2D every update",
                          "step_timing": "CUDA synchronize before and after H2D+forward+backward+optimizer",
                          "excluded": ["video/face decoding", "face tracking/cropping", "MFCC extraction",
                                       "data augmentation", "disk/network data loading", "evaluation",
                                       "checkpoint saving", "host input generation", "per-step logging"]},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "pid": os.getpid(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                        "git_revision": command_output(["git", "rev-parse", "HEAD"], args.root),
                        "git_worktree_status": command_output(["git", "status", "--short"], args.root)},
        "phase_seconds": {}, "optimizer_updates": 0, "steps": [],
        "memory_before": memory_snapshot(),
        "gpu_monitor_units": {"memory.used": "MiB (total device usage, includes other processes)",
                              "utilization.gpu": "percent", "power.draw": "watts",
                              "temperature.gpu": "degrees Celsius", "clocks.sm": "MHz",
                              "clocks.mem": "MHz", "sample_interval_sec": 1},
    }
    monitor.start()
    exit_code = 0
    try:
        phase = time.perf_counter()
        import numpy as np
        import torch
        sys.path.insert(0, str(args.root))
        from ASD import ASD
        result["phase_seconds"]["imports"] = time.perf_counter() - phase
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; benchmark requires the actual GPU")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        result["environment"].update({
            "numpy": np.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu_name": properties.name,
            "gpu_total_bytes": properties.total_memory, "device_index": device,
            "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        })
        phase = time.perf_counter()
        model = ASD(lr=args.lr)
        model.train()
        torch.cuda.synchronize()
        result["phase_seconds"]["model_construct"] = time.perf_counter() - phase
        phase = time.perf_counter()
        result["environment"]["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        load_result = model.load_state_dict(state, strict=True)
        torch.cuda.synchronize()
        result["checkpoint_load"] = {"strict": True, "missing_keys": load_result.missing_keys,
                                     "unexpected_keys": load_result.unexpected_keys,
                                     "state_tensors": len(state)}
        del state
        result["phase_seconds"]["checkpoint_load_and_hash"] = time.perf_counter() - phase
        phase = time.perf_counter()
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        initial = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
        if any(parameter.dtype != torch.float32 for parameter in trainable.values()):
            raise RuntimeError("Expected every trainable parameter to be float32")
        optimized_ids = {id(parameter) for group in model.optim.param_groups for parameter in group["params"]}
        if not all(id(parameter) in optimized_ids for parameter in trainable.values()):
            raise RuntimeError("Some trainable parameters are absent from the optimizer")
        result["trainable_parameter_count"] = sum(parameter.numel() for parameter in trainable.values())
        result["trainable_tensor_count"] = len(trainable)
        audio_host = torch.randn(args.batch_size, args.frames * 4, 13).pin_memory()
        visual_host = torch.randint(0, 256, (args.batch_size, args.frames, 112, 112),
                                    dtype=torch.int32).float().pin_memory()
        labels_host = torch.randint(0, 2, (args.batch_size * args.frames,), dtype=torch.int64).pin_memory()
        result["pinned_input_bytes"] = sum(t.numel() * t.element_size()
                                           for t in (audio_host, visual_host, labels_host))
        result["phase_seconds"]["host_inputs_and_parameter_baseline"] = time.perf_counter() - phase
        result["memory_before_training"] = memory_snapshot()
        torch.cuda.reset_peak_memory_stats()
        monitor.phase = "training"
        training_start = time.perf_counter()
        for step in range(1, args.steps + 1):
            torch.cuda.synchronize()
            started = time.perf_counter()
            audio = audio_host.to(device="cuda", non_blocking=True)
            visual = visual_host.to(device="cuda", non_blocking=True)
            labels = labels_host.to(device="cuda", non_blocking=True)
            model.zero_grad()
            audio_embed = model.model.forward_audio_frontend(audio)
            visual_embed = model.model.forward_visual_frontend(visual)
            outputs_av = model.model.forward_audio_visual_backend(audio_embed, visual_embed)
            outputs_v = model.model.forward_visual_backend(visual_embed)
            loss_av, _, _, _ = model.lossAV(outputs_av, labels)
            loss_v = model.lossV(outputs_v, labels)
            loss = loss_av + 0.5 * loss_v
            loss.backward()
            model.optim.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            result["optimizer_updates"] += 1
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                result["all_losses_finite"] = False
                raise RuntimeError("Non-finite training loss at update " + str(step))
            memory = memory_snapshot()
            row = {"step": step, "warmup": step <= args.warmup, "seconds": elapsed,
                   "frames_per_second": args.batch_size * args.frames / elapsed,
                   "loss": loss_value, "loss_av": loss_av.item(), "loss_v": loss_v.item(),
                   "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                   "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                   "process_rss_kib": memory["process_kib"]["VmRSS"],
                   "process_hwm_kib": memory["process_kib"]["VmHWM"],
                   "wsl_available_kib": memory["wsl_kib"]["MemAvailable"],
                   "process_swap_kib": memory["process_kib"]["VmSwap"],
                   "wsl_swap_free_kib": memory["wsl_kib"]["SwapFree"],
                   "swap_in_pages": memory["swap_pages"].get("pswpin"),
                   "swap_out_pages": memory["swap_pages"].get("pswpout")}
            result["steps"].append(row)
            if step % 20 == 0 or step == args.steps:
                print(f"update {step}/{args.steps}: {elapsed:.4f} s, "
                      f"{row['frames_per_second']:.1f} frames/s, loss={loss_value:.6f}, "
                      f"CUDA peak={torch.cuda.max_memory_allocated() / 2**30:.3f} GiB, "
                      f"RSS={(row['process_rss_kib'] or 0) / 2**20:.3f} GiB", flush=True)
            # Release graph references between steps, as the upstream training loop does.
            del audio, visual, labels, audio_embed, visual_embed, outputs_av, outputs_v, loss, loss_av, loss_v
        result["phase_seconds"]["training_loop_wall"] = time.perf_counter() - training_start
        monitor.phase = "final_metrics"
        phase = time.perf_counter()
        delta_squared = 0.0
        initial_squared = 0.0
        maximum_delta = 0.0
        changed_elements = 0
        changed_tensors = 0
        for name, parameter in trainable.items():
            delta = parameter.detach().cpu().double() - initial[name].double()
            changed = int(torch.count_nonzero(delta).item())
            changed_elements += changed
            changed_tensors += int(changed > 0)
            delta_squared += delta.square().sum().item()
            initial_squared += initial[name].double().square().sum().item()
            maximum_delta = max(maximum_delta, delta.abs().max().item())
        result["parameter_delta"] = {"l2": math.sqrt(delta_squared),
                                     "relative_l2": math.sqrt(delta_squared / initial_squared)
                                     if initial_squared else None,
                                     "maximum_absolute": maximum_delta,
                                     "changed_elements": changed_elements, "changed_tensors": changed_tensors}
        optimizer_steps = [int(state["step"].item()) for state in model.optim.state.values() if "step" in state]
        result["optimizer_state_steps"] = {"min": min(optimizer_steps), "max": max(optimizer_steps),
                                           "state_count": len(optimizer_steps)}
        result["cuda_memory"] = {"max_allocated_bytes": torch.cuda.max_memory_allocated(),
                                  "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                                  "final_allocated_bytes": torch.cuda.memory_allocated(),
                                  "final_reserved_bytes": torch.cuda.memory_reserved()}
        result["memory_after_training"] = memory_snapshot()
        result["phase_seconds"]["final_metrics"] = time.perf_counter() - phase
        all_times = [row["seconds"] for row in result["steps"]]
        steady_times = all_times[args.warmup:]
        result["timing"] = {"all_updates": summarize(all_times, np),
                             "warmup_updates": summarize(all_times[:args.warmup], np),
                             "steady_updates": summarize(steady_times, np),
                             "steady_frames_per_second": args.batch_size * args.frames / statistics.mean(steady_times)}
        if changed_elements == 0 or min(optimizer_steps) != args.steps or max(optimizer_steps) != args.steps:
            raise RuntimeError("Optimizer update verification failed")
        result["all_losses_finite"] = True
        result["status"] = "complete"
    except Exception as error:
        exit_code = 1
        result["status"] = "failed"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        monitor.stop()
        if "torch" in locals() and torch.cuda.is_available():
            result.setdefault("cuda_memory", {
                "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                "final_allocated_bytes": torch.cuda.memory_allocated(),
                "final_reserved_bytes": torch.cuda.memory_reserved()})
        result["gpu_samples"] = monitor.samples
        result["gpu_monitor_errors"] = monitor.errors
        summaries = {}
        for sample in monitor.samples:
            if sample["phase"] != "training":
                continue
            gpu = str(sample.get("index", "unknown"))
            summaries.setdefault(gpu, {})
            for key in monitor.fields[2:]:
                value = sample.get(key)
                if isinstance(value, (int, float)):
                    summaries[gpu].setdefault(key, []).append(value)
        result["training_gpu_summary"] = {
            gpu: {key: {"samples": len(values), "mean": statistics.mean(values),
                        "min": min(values), "max": max(values)}
                  for key, values in metrics.items()} for gpu, metrics in summaries.items()}
        result["memory_final"] = memory_snapshot()
        result["phase_seconds"]["script_wall_before_output"] = time.perf_counter() - SCRIPT_START
        write_csv(args.output_dir / "steps.csv", result["steps"])
        write_csv(args.output_dir / "gpu-samples.csv", monitor.samples)
        report = args.output_dir / "benchmark.json"
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"status={result['status']}; optimizer_updates={result['optimizer_updates']}; report={report}", flush=True)
        if result["status"] == "complete":
            print(json.dumps({"timing": result["timing"], "cuda_memory": result["cuda_memory"],
                              "parameter_delta": result["parameter_delta"]}, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
