import argparse

import numpy as np

import torch

import tvm
from tvm import autotvm, relay, testing
from tvm.contrib import ndk, rpc, utils
import tvm.contrib.debugger.debug_executor as debug_executor
import tvm.contrib.graph_executor as runtime


from baseline.utils import quantize_torch 
from baseline.model_archive import *

import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--tuning-records", default="resnet18.json")
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--target", default="x86")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9190, type=int)
    parser.add_argument("--key", default="pixel4")
    args = parser.parse_args()

    assert args.target in ["x86", "arm"]

    os.environ["TVM_NUM_THREADS"] = str(args.num_threads)

    model, input_tensors = globals()[args.model](args.quantize)
    model = model.eval()
    if args.quantize:
        quantize_torch(model, input_tensors)
    scripted_model = torch.jit.trace(model, input_tensors).eval()

    input_infos = [
        (i.debugName().split('.')[0], i.type().sizes())
        for i in list(scripted_model.graph.inputs())[1:]
    ]
    
    if args.quantize:
        mod, params = tvm.relay.frontend.from_pytorch(
            scripted_model, input_infos, keep_quantized_weight=True)
    else:
        mod, params = tvm.relay.frontend.from_pytorch(
            scripted_model, input_infos)


    if args.target == "x86":
        target = "llvm -mcpu=core-avx2"
    elif args.target == "arm":
        target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"

    with autotvm.apply_history_best(args.tuning_records):
        with tvm.transform.PassContext(opt_level=3, config={}):
            lib = relay.build(mod, target=target, params=params)

    if args.target == "x86":
        ctx = tvm.device(str(target), 0)
        m = runtime.GraphModule(lib["default"](ctx))
    elif args.target == "arm":
        libname = "model.so"
        temp = utils.tempdir()
        libpath = temp.relpath(libname)
        lib.export_library(libpath, ndk.create_shared)
        remote = rpc.connect_tracker(args.host, args.port).request(args.key)
        remote.upload(libpath)
        rlib = remote.load_module(libname)
        ctx = remote.cpu(0)
        m = runtime.GraphModule(rlib["default"](ctx))

    for input_info, input_tensor in zip(input_infos, input_tensors):
        m.set_input(
            input_info[0],
            tvm.nd.array(input_tensor.cpu().numpy(), ctx)
        )
    m.set_input(**lib.get_params())
    print(m.benchmark(ctx, number=1, repeat=600))
