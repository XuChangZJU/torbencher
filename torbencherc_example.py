from src.torbencherc import torbencherc
from multiprocessing import freeze_support

config = {
    "out_dir": "./results",
    "seed": 1234567890,
    "devices": [
        "cpu",
        "cuda",
        # "mps",
        # other device names...
    ],
    "test_modules": [
        "torch",                                # From `TorchCompileTestCase` need checking on non-windows
        "torch.nn",                             # Can B run through, Plz fixing `Failed`
        "torch.nn.functional",                  # Can B run through, Plz fixing `Failed`
        "torch.special",                        # Can B run through, Plz fixing `Failed`
        # "torch.autograd",                     # From `TorchAutogradFunctionNestediofunctionTestCase` need checking on cuda
        # "torch.utils.checkpoint",             # Well Done
        # "torch.export",                       # Windows not support Compile, and I can do nothing
        # "torch.profiler",                     # Both support, but can't be judged, no need to test
        # "torch.profiler.itt",                 # Well Done
        "torch.testing",                        # Well Done
        # "torch.nn.init",                      # CPU passed all, CUDA passed half
        "torch.fft",                            # Well Done
        # "torch.nn.parallel",                  # Need at least 2 GPUs, can't run on Personal Computer
        # "torch.nn.utils",                     # 'TorchNnUtilsClipgradvalueTestCase' stucked, and other `Failed` to fix
        # "torch.nn.utils.parametrize",         # 'TorchNnUtilsParametrizeRemoveparametrizationsTestCase' is a huge problem
        # "torch.nn.utils.prune",               # hard to say, no one can run
        # "torch.nn.utils.parametrizations",    # 'TorchNnUtilsParametrizationsOrthogonalTestCase' is a huge problem
        "torch.nn.utils.stateless",             # Only one, Well Done.
        # "torch.nn.utils.rnn",                 # Needs fixing from 'TorchNnUtilsRnnPacksequenceTestCase'
        # "torch.nn.modules.module",            # Abstruct Object, Hard to fix
        # "torch.nn.modules.lazy",              # Well Done, but device can only B on "device", non-cpu
        # "torch.autograd.forward_ad",          # Well Done
        # "torch.autograd.gradcheck",           # Empty, needs to be dived into?
        # "torch.autograd.graph",               # Only `TorchAutogradGraphRegistermultigradhookTestCase` needs fixing****
        # "torch.autograd.Function",            # Well Done on CPU, None Done on CUDA*****
        "torch.autograd.profiler_util",         # Well Done
        "torch.autograd.grad_mode",             # Well Done
        # "torch.autograd.profiler",            # From 'TorchAutogradProfilerLoadnvprofTestCase' need to be fixed
        "torch.autograd.functional",            # Well Done
        # "torch.autograd.graph.Node",          # From 'TorchAutogradGraphNodeRegisterprehookTestCase' need to be fixed
        # "torch.autograd.Function.FunctionCtx",# Well Done on CPU, Not Done on CUDA at least `TorchAutogradFunctionFunctionctxMarknondifferentiableTestCase`
        # "torch.autograd.profiler.profile",    # Can B run through, Plz fixing `Failed`
        # ***************************************************
        # "torch.optim",                        # No one can run on cuda
        # "torch.optim.Optimizer",              # No one can run on cuda
        # "torch.optim.lrscheduler",            # No one can run on cuda
        # "torch.utils.data",                   # 'TorchUtilsDataGetworkerinfoTestCase' stucked, hard to fix
        # "torch.utils.data._utils",            # Empty, needs to be dived into?
        # "torch.utils.data.distributed",       # 'TorchUtilsDataDistributedDistributedsamplerTestCase' can only run on cuda
        # "torch.utils.data.utils",             # Empty, needs to be dived into?
        # "torch.utils.data._utils.collate",    # Only one, but Well Done
        # "torch.utils.data.utils.collate",     # Only one, but Well Done(Doubt this the same as the one above)
    ],
    "format": "csv",
    "num_epoch": 1,
    "name_spec": "timestamp"
}
freeze_support()
bencher = torbencherc(config)
result = bencher.run()
print(result)
