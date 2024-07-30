from src import bencherDebugger

modules = [
    "torch.nn.functional",
    "torch.optim",
    "torch.special",
    # "torch.random", # no testcases
    # "torch.utils.cpp_extension", # skip
    "torch.utils.data",
    # "torch.xpu", # skip
    # "torch.mps", # skip
    "torch.jit",
    "torch.utils",
    "torch.distributions",
    "torch.autograd",
    "torch",
    # "torch.onnx", # no testcases
    # "torch.cuda", # skip
    "torch.linalg",
    # "torch.amp", # no testcases
    "torch.nn",
    "torch.utils.mobile_optimizer",
    # "torch.distributed", # no testcases
    "torch.utils.checkpoint",
    "torch.Tensor",
    "torch.export",
    "torch.profiler",
    "torch.backends",
    # "torch.fx", # skip
    "torch.cpu",
    # "torch.hub", # skip
    "torch.testing",
    # "torch.masked", # skip
    # "torch.utils.tensorboard", # skip
    "torch.nn.init",
    "torch.fft",
    "torch.autograd"
]

debugger = bencherDebugger(
    {
        "seed": 1234567890,
        "devices": ["cpu"],
        "test_modules": modules,
        "format": "json",
        "num_epoches": 5,
        "including_success": False
    }
)
result = debugger.run()
print("Done")
