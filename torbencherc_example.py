from src.torbencherc import torbencherc

config = {
    "out_dir": "./results",
    "seed": 1234567890,
    "devices": [
        "cuda",
        # other device names...
    ],
    "test_modules": [
        "torch.nn.functional",
        "torch.optim",
        "torch.special",
        "torch.utils.data",
        "torch.utils",
        "torch.autograd",
        "torch",
        "torch.nn",
        "torch.utils.mobile_optimizer",
        "torch.utils.checkpoint",
        "torch.export",
        "torch.profiler",
        "torch.testing",
        "torch.nn.init",
        "torch.fft",
        "torch.autograd",

        "torch.nn.parallel",
        "torch.nn.utils",
        "torch.nn.modules",
        "torch.nn.functional",
        "torch.nn.utils.parametrize",
        "torch.nn.utils.prune",
        "torch.nn.utils.parametrizations",
        "torch.nn.utils.stateless",
        "torch.nn.utils.rnn",
        "torch.nn.modules.module",
        "torch.nn.modules.lazy",
        "torch.autograd.forward_ad",
        "torch.autograd.gradcheck",
        "torch.autograd.graph",
        "torch.autograd.Function",
        "torch.autograd.profiler_util",
        "torch.autograd.grad_mode",
        "torch.autograd.profiler",
        "torch.autograd.functional",
        "torch.autograd.graph.Node",
        "torch.autograd.Function.FunctionCtx",
        "torch.autograd.profiler.profile",
        "torch.optim.Optimizer",
        "torch.optim.lrscheduler",
        "torch.utils.data",
        "torch.utils.data._utils",
        "torch.utils.data.distributed",
        "torch.utils.data.utils",
        "torch.utils.data._utils.collate",
        "torch.utils.data.utils.collate",
        "torch.profiler.itt",
        "torch.export.dynamic_shapes",
        "torch.export.graph_signature"
    ],
    "format": "csv",
    "num_epoch": 1,
    "name_spec": "timestamp"
}

bencher = torbencherc(config)
result = bencher.run()
print(result)
