"""
Reference ML inference workloads used both during original data collection
and by `cetp measure` to benchmark the current machine directly. Extracted
verbatim from the data-collection shell history (identical across every
cloud tier used during collection; only iteration counts differed per
machine, which is why calibration is per-machine, not shared).
"""

import torch
import torchvision.models as tv_models
from transformers import AutoTokenizer, AutoModel
import psutil

torch.set_num_threads(psutil.cpu_count() or 4)

_resnet18 = None
_resnet50 = None
_mobilenet = None
_distilbert_tokenizer = None
_distilbert_model = None


def _load_resnet18():
    global _resnet18
    if _resnet18 is None:
        _resnet18 = tv_models.resnet18(weights=None)
        _resnet18.eval()
    return _resnet18


def _load_resnet50():
    global _resnet50
    if _resnet50 is None:
        _resnet50 = tv_models.resnet50(weights=None)
        _resnet50.eval()
    return _resnet50


def _load_mobilenet():
    global _mobilenet
    if _mobilenet is None:
        _mobilenet = tv_models.mobilenet_v2(weights=None)
        _mobilenet.eval()
    return _mobilenet


def _load_distilbert():
    global _distilbert_tokenizer, _distilbert_model
    if _distilbert_model is None:
        _distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        _distilbert_model = AutoModel.from_pretrained("distilbert-base-uncased")
        _distilbert_model.eval()
    return _distilbert_tokenizer, _distilbert_model


def run_resnet18(batch_size: int, num_iterations: int):
    model = _load_resnet18()
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        for _ in range(num_iterations):
            model(x)


def run_resnet50(batch_size: int, num_iterations: int):
    model = _load_resnet50()
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        for _ in range(num_iterations):
            model(x)


def run_mobilenet(batch_size: int, num_iterations: int):
    model = _load_mobilenet()
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        for _ in range(num_iterations):
            model(x)


def run_distilbert(batch_size: int, num_iterations: int):
    tokenizer, model = _load_distilbert()
    texts = ["This is a sample sentence for benchmarking inference latency."] * batch_size
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        for _ in range(num_iterations):
            model(**inputs)


WORKLOAD_FUNCTIONS = {
    "resnet18": run_resnet18,
    "resnet50": run_resnet50,
    "mobilenet": run_mobilenet,
    "distilbert": run_distilbert,
}
