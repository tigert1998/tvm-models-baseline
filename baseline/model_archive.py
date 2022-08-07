import os
import os.path as osp

import torch
import torchvision


def bert_half():
    from blink_mm.networks.seq_models.bert import BERT
    from baseline.utils import get_bert_sample_input

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    bert = BERT(2, "cpu", num_hidden_layers=6, torchscript=True)
    input_ids, attention_mask, token_type_ids = get_bert_sample_input()
    return bert, (input_ids, attention_mask, token_type_ids)


def bert_base():
    from blink_mm.networks.seq_models.bert import BERT
    from baseline.utils import get_bert_sample_input

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    bert = BERT(2, "cpu", torchscript=True)
    input_ids, attention_mask, token_type_ids = get_bert_sample_input()
    return bert, (input_ids, attention_mask, token_type_ids)


def resnet18():
    dummy_input = torch.randn(1, 3, 224, 224)
    return torchvision.models.resnet18(pretrained=True), (dummy_input,)


def resnet50():
    dummy_input = torch.randn(1, 3, 224, 224)
    return torchvision.models.resnet50(pretrained=True), (dummy_input,)


def resnet20():
    import blink_mm.networks.pytorch_resnet_cifar10.resnet as resnet
    from blink_mm.expers.train_cifar import strip_state_dict_prefix
    dummy_input = torch.randn(1, 3, 32, 32)
    model = resnet.resnet20()
    model.load_state_dict(strip_state_dict_prefix(torch.load(
        osp.join(osp.dirname(resnet.__file__),
                 "pretrained_models/resnet20-12fca82f.th")
    )))
    return model, (dummy_input,)


def resnet32():
    import blink_mm.networks.pytorch_resnet_cifar10.resnet as resnet
    from blink_mm.expers.train_cifar import strip_state_dict_prefix
    dummy_input = torch.randn(1, 3, 32, 32)
    model = resnet.resnet32()
    model.load_state_dict(strip_state_dict_prefix(torch.load(
        osp.join(osp.dirname(resnet.__file__),
                 "pretrained_models/resnet32-d509ac18.th")
    )))
    return model, (dummy_input,)
