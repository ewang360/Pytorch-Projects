# benchmark_pytorch.py
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torchvision
import torch.utils.benchmark as benchmark


@torch.no_grad()
def measure_time_host(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_repeats: int = 100,
    num_warmups: int = 10,
    continuous_measure: bool = True,
) -> float:

    for _ in range(num_warmups):
        _ = model.forward(input_tensor)

    elapsed_time_ms = 0

    if continuous_measure:
        start = timer()
        for _ in range(num_repeats):
            _ = model.forward(input_tensor)
        end = timer()
        elapsed_time_ms = (end - start) * 1000

    else:
        for _ in range(num_repeats):
            start = timer()
            _ = model.forward(input_tensor)
            end = timer()
            elapsed_time_ms += (end - start) * 1000

    return elapsed_time_ms / num_repeats


@torch.no_grad()
def run_inference(model: nn.Module,
                  input_tensor: torch.Tensor) -> torch.Tensor:

    return model.forward(input_tensor)


def main() -> None:

    num_warmups = 100
    num_repeats = 1000
    input_shape = (1, 3, 224, 224)

    model = torchvision.models.resnet18(pretrained=False)
    # model = nn.Conv2d(in_channels=input_shape[1],
    #                   out_channels=256,
    #                   kernel_size=(5, 5))

    model.eval()

    # Input tensor
    input_tensor = torch.rand(input_shape)

    print("Latency Measurement Using CPU Timer...")
    for continuous_measure in [True, False]:
        try:
            latency_ms = measure_time_host(
                model=model,
                input_tensor=input_tensor,
                num_repeats=num_repeats,
                num_warmups=num_warmups,
                continuous_measure=continuous_measure,
            )
            print(f"|"
                    f"Continuous Measurement: {continuous_measure!s:5}| "
                    f"Latency: {latency_ms:.5f} ms| ")
        except Exception as e:
            print(f"|"
                    f"Continuous Measurement: {continuous_measure!s:5}| "
                    f"Latency: N/A     ms| ")

    print("Latency Measurement Using PyTorch Benchmark...")
    num_threads = 1
    timer = benchmark.Timer(stmt="run_inference(model, input_tensor)",
                            setup="from __main__ import run_inference",
                            globals={
                                "model": model,
                                "input_tensor": input_tensor
                            },
                            num_threads=num_threads,
                            label="Latency Measurement",
                            sub_label="torch.utils.benchmark.")

    profile_result = timer.timeit(num_repeats)
    # https://pytorch.org/docs/stable/_modules/torch/utils/benchmark/utils/common.html#Measurement
    print(f"Latency: {profile_result.mean * 1000:.5f} ms")


if __name__ == "__main__":

    main()
