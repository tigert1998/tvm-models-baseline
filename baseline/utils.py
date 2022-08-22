from PIL import Image

from tvm import autotvm, relay
from tvm.autotvm.tuner import XGBTuner
from tvm.contrib.download import download_testdata

import torch
from torchvision import transforms


def get_imagenet_preprocess():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def get_imagenet_sample_image():
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    return Image.open(img_path)


def get_bert_sample_input():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    dic = tokenizer(text, padding="max_length", truncation=True, max_length=64)
    input_ids = torch.tensor(dic["input_ids"])[None, :]
    attention_mask = torch.tensor(dic["attention_mask"])[None, :]
    token_type_ids = torch.tensor(dic["token_type_ids"])[None, :]
    return input_ids, attention_mask, token_type_ids


def quantize_torch(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp[0])
    torch.quantization.convert(model, inplace=True)

def quantize(mod, params, data_aware, **kwargs):
    qconfig_kwargs = {
        "skip_dense_layer": False,
        "skip_conv_layers": []
    }
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max", **qconfig_kwargs):
            mod = relay.quantize.quantize(
                mod, params, dataset=kwargs["calibrate_dataset"]())
    else:
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0, **qconfig_kwargs):
            mod = relay.quantize.quantize(mod, params)
    return mod


def tune_network(mod, params, target, tuning_option):
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params)

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d, %s] " % (i + 1, len(tasks), task.name)
        tuner = XGBTuner(task, loss_type="rank", feature_type="curve")
        tuner.tune(
            n_trial=min(tuning_option["n_trial"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(
                    tuning_option["n_trial"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )
