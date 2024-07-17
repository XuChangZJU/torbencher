from torbencher import benchdebugger

modules = [
"torch.nn.functional",
"torch.optim",
"torch.special",
"torch.random",
"torch.utils.cpp_extension",
"torch.utils.data",
# "torch.xpu",
# "torch.mps",
"torch.jit",
"torch.utils",
"torch.distributions",
"torch.autograd",
"torch",
"torch.onnx",
# "torch.cuda",
"torch.linalg",
"torch.amp",
"torch.nn",
"torch.utils.mobile_optimizer",
"torch.distributed",
"torch.utils.checkpoint",
"torch.Tensor",
"torch.export",
"torch.profiler",
"torch.backends",
"torch.fx",
"torch.cpu",
# "torch.hub",
"torch.testing",
"torch.masked",
"torch.utils.tensorboard",
"torch.nn.init",
"torch.fft",
"torch.autograd"
]

debugger = benchdebugger(
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
