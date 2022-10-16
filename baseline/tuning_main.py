import os
import argparse

import torch
import torch.utils.dlpack

import tvm
from tvm import autotvm, auto_scheduler
import tvm.relay

from baseline.utils import quantize, tune_network, tune_network_auto_scheduler
from baseline.model_archive import MODEL_ARCHIVE

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--tuning-records", default="resnet18.json")
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--tuner", default="autotvm",
                        choices=["autotvm", "auto_scheduler"])
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--target", default="x86", choices=["x86", "arm"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9190, type=int)
    parser.add_argument("--key", default="pixel4")
    args = parser.parse_args()

    os.environ["TVM_NUM_THREADS"] = str(args.num_threads)

    model_info = MODEL_ARCHIVE[args.model]

    model = model_info["model"]()
    input_tensors = model_info["input"]
    scripted_model = torch.jit.trace(model, input_tensors).eval()

    input_infos = [
        (i.debugName().split('.')[0], i.type().sizes())
        for i in list(scripted_model.graph.inputs())[1:]
    ]
    mod, params = tvm.relay.frontend.from_pytorch(
        scripted_model, input_infos)

    if args.quantize:
        mod = quantize(mod, params, False)

    if args.target == "x86":
        target = "llvm -mcpu=cascadelake"
        if args.tuner == "autotvm":
            measure_option = autotvm.measure_option(
                builder="local", runner="local"
            )
        elif args.tuner == "auto_scheduler":
            builder = auto_scheduler.LocalBuilder()
            runner = auto_scheduler.LocalRunner(
                repeat=10, enable_cpu_cache_flush=True
            )
    elif args.target == "arm":
        target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
        if args.tuner == "autotvm":
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="ndk", timeout=60),
                runner=autotvm.RPCRunner(
                    args.key, args.host, args.port)
            )
        elif args.tuner == "auto_scheduler":
            builder = auto_scheduler.LocalBuilder(build_func="ndk", timeout=60)
            runner = auto_scheduler.RPCRunner(
                args.key,
                host=args.host,
                port=args.port,
                repeat=10,
                min_repeat_ms=200,
                enable_cpu_cache_flush=True,
            )

    if args.tuner == "autotvm":
        tuning_option = {
            "n_trial": 1500,
            "early_stopping": None,
            "measure_option": measure_option,
            "tuning_records": args.tuning_records,
        }

        tune_network(mod, params, target, tuning_option)
    elif args.tuner == "auto_scheduler":
        tuning_option = auto_scheduler.TuningOptions(
            num_measure_trials=15000,
            builder=builder,
            runner=runner,
            measure_callbacks=[
                auto_scheduler.RecordToFile(args.tuning_records)],
        )

        tune_network_auto_scheduler(mod, params, target, tuning_option)
