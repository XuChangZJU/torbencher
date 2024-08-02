from src.torbencherc import torbencherc
from multiprocessing import freeze_support

config = {
    "out_dir": "./results",
    "seed": 1234567890,
    "devices": [
        "cpu",
        "cuda",
        # other device names...
    ],
    "test_modules": [
         # "torch.nn.functional",           # Can B run through, Plz fixing `Failed`
        # "torch.optim",                    # No one can run on cuda
        # "torch.special",                  # Can B run through, Plz fixing `Failed`
        # "torch.autograd",                 # From `TorchAutogradFunctionNestediofunctionTestCase` need checking on cuda
        # "torch",                          # From `TorchCompileTestCase` need checking on non-windows
        # "torch.nn",                       # Can B run through, Plz fixing `Failed`
        # "torch.utils.mobile_optimizer",   # Do not support CUDA.
        # "torch.utils.checkpoint",         # Well Done
        # "torch.export",                   # Windows not support Compile, and I can do nothing
        # "torch.profiler",                 # Both support, but can't be judged, no need to test
        # "torch.backends",                 # Empty, needs to be dived into.
        "torch.cpu",
        # "torch.testing",
        # "torch.nn.init",
        # "torch.fft",
        # "torch.autograd",
        #
        # "torch.nn.parallel",
        # "torch.nn.utils",
        # "torch.nn.modules",
        # "torch.nn.functional",
        # "torch.nn.utils.parametrize",
        # "torch.nn.utils.prune",
        # "torch.nn.utils.parametrizations",
        # "torch.nn.utils.stateless",
        # "torch.nn.utils.rnn",
        # "torch.nn.modules.module",
        # "torch.nn.modules.lazy",
        # "torch.autograd.forward_ad",
        # "torch.autograd.gradcheck",
        # "torch.autograd.graph",
        # "torch.autograd.torch",
        # "torch.autograd.Function",
        # "torch.autograd.profiler_util",
        # "torch.autograd.grad_mode",
        # "torch.autograd.profiler",
        # "torch.autograd.functional",
        # "torch.autograd.graph.Node",
        # "torch.autograd.torch.Tensor",
        # "torch.autograd.Function.FunctionCtx",
        # "torch.autograd.profiler.profile",
        # "torch.backends.opt_einsum",
        # "torch.backends.mha",
        # "torch.backends.nnpack",
        # "torch.backends.mps",
        # "torch.backends.cpu",
        # "torch.backends.cuda",
        # "torch.backends.mkl",
        # "torch.backends.mkldnn",
        # "torch.backends.openmp",
        # "torch.backends.cudnn",
        # "torch.backends.cuda.cufftplancache",
        # "torch.backends.cuda.cufft_plan_cache",
        # "torch.backends.cuda.cufftplancache.torch",
        # "torch.backends.cuda.cufftplancache.torch.backends",
        # "torch.backends.cuda.cufftplancache.torch.backends.cuda",
        # "torch.backends.cuda.cufft_plan_cache.torch",
        # "torch.backends.cuda.cufft_plan_cache.torch.backends",
        # "torch.backends.cuda.cufft_plan_cache.torch.backends.cuda",
        # "torch.optim.Optimizer",
        # "torch.optim.lrscheduler",
        # "torch.utils.data",
        # "torch.utils.data._utils",
        # "torch.utils.data.distributed",
        # "torch.utils.data.torch",
        # "torch.utils.data.utils",
        # "torch.utils.data._utils.collate",
        # "torch.utils.data.utils.collate",
        # "torch.profiler.itt",
        # "torch.export.dynamic_shapes",
        # "torch.export.graph_signature"
    ],
    "format": "csv",
    "num_epoch": 1,
    "name_spec": "timestamp"
}
freeze_support()
bencher = torbencherc(config)
result = bencher.run()
print(result)
