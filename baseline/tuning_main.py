import argparse

import torch
import torch.utils.dlpack

import tvm
from tvm import autotvm
import tvm.relay

from baseline.utils import quantize, tune_network
from baseline.model_archive import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--tuning-records", default="resnet18.json")
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--target", default="x86")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9190)
    parser.add_argument("--key", default="pixel4")
    args = parser.parse_args()

    assert args.target in ["x86", "arm"]

    os.environ["TVM_NUM_THREADS"] = str(args.num_threads)

    model, input_tensors = globals()[args.model]()
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
        target = "llvm -mcpu=core-avx2"
        measure_option = autotvm.measure_option(
            builder="local", runner="local"
        )
    elif args.target == "arm":
        target = "llvm -mtriple=aarch64-arm-none-eabi"
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="ndk"),
            runner=autotvm.RPCRunner(args.key, args.host, args.port)
        )

    tuning_option = {
        "n_trial": 1500,
        "early_stopping": None,
        "measure_option": measure_option,
        "tuning_records": args.tuning_records,
    }

    tune_network(mod, params, target, tuning_option)
