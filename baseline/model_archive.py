import torch
import torchvision


def resnet18():
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.to("cpu")
    model.eval()
    return model


def resnet50():
    model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    model.to("cpu")
    model.eval()
    return model


MODEL_ARCHIVE = {
    "resnet18": {
        "model": resnet18,
        "input": (torch.randn(1, 3, 224, 224),),
    },
    "resnet50": {
        "model": resnet50,
        "input": (torch.randn(1, 3, 224, 224),),
    },
}
