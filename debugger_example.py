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
    # "torch.jit",
    "torch.utils",
    "torch.distributions",
    "torch.autograd",
    "torch",
    # "torch.onnx", # no testcases
    # "torch.cuda", # skip
    # "torch.linalg",
    # "torch.amp", # no testcases
    "torch.nn",
    "torch.utils.mobile_optimizer",
    # "torch.distributed", # no testcases
    "torch.utils.checkpoint",
    # "torch.Tensor",
    "torch.export",
    "torch.profiler",
    # "torch.backends", # skip
    # "torch.fx", # skip
    "torch.cpu",
    # "torch.hub", # skip
    "torch.testing",
    # "torch.masked", # skip
    # "torch.utils.tensorboard", # skip
    "torch.nn.init",
    "torch.fft",
    "torch.autograd",
    # "torch.linalg",

    # "torch.nn",
    # "torch.cpu",
    # "torch.autograd",
    # "torch.fx",
    # "torch.cuda",
    # "torch.backends",
    # "torch.optim",
    # "torch.utils",
    # "torch.special",
    # "torch.testing",
    # "torch.jit",
    # "torch.fft",
    # "torch.profiler",
    # "torch.Tensor",
    # "torch.export",
    # "torch.hub",
    # "torch.distributions",
    "torch.nn.init",
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
    "torch.cpu.amp",
    "torch.autograd.forward_ad",
    "torch.autograd.gradcheck",
    "torch.autograd.graph",
    "torch.autograd.torch",
    "torch.autograd.Function",
    "torch.autograd.profiler_util",
    "torch.autograd.grad_mode",
    "torch.autograd.profiler",
    "torch.autograd.functional",
    "torch.autograd.graph.Node",
    "torch.autograd.torch.Tensor",
    "torch.autograd.Function.FunctionCtx",
    "torch.autograd.profiler.profile",
    # "torch.cuda.amp",
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
    "torch.optim.Optimizer",
    "torch.optim.lrscheduler",
    "torch.utils.mobile_optimizer",
    "torch.utils.checkpoint",
    "torch.utils.cpp_extension",
    "torch.utils.tensorboard",
    "torch.utils.data",
    "torch.utils.tensorboard.writer",
    "torch.utils.data._utils",
    "torch.utils.data.distributed",
    "torch.utils.data.utils",
    "torch.utils.data._utils.collate",
    "torch.utils.data.utils.collate",
    "torch.profiler.itt",
    "torch.export.dynamic_shapes",
    "torch.export.graph_signature",
    # "torch.distributions.beta",
    # "torch.distributions.gamma",
    # "torch.distributions.negative_binomial",
    # "torch.distributions.von_mises",
    # "torch.distributions.dirichlet",
    # "torch.distributions.bernoulli",
    # "torch.distributions.transformed_distribution",
    # "torch.distributions.one_hot_categorical",
    # "torch.distributions.wishart",
    # "torch.distributions.relaxed_categorical",
    # "torch.distributions.weibull",
    # "torch.distributions.kl",
    # "torch.distributions.kumaraswamy",
    # "torch.distributions.laplace",
    # "torch.distributions.exponential",
    # "torch.distributions.binomial",
    # "torch.distributions.uniform",
    # "torch.distributions.multinomial",
    # "torch.distributions.chi2",
    # "torch.distributions.distribution",
    # "torch.distributions.geometric",
    # "torch.distributions.log_normal",
    # "torch.distributions.pareto",
    # "torch.distributions.cauchy",
    # "torch.distributions.half_normal",
    # "torch.distributions.categorical",
    # "torch.distributions.continuous_bernoulli",
    # "torch.distributions.mixture_same_family",
    # "torch.distributions.fishersnedecor",
    # "torch.distributions.lowrank_multivariate_normal",
    # "torch.distributions.studentT",
    # "torch.distributions.independent",
    # "torch.distributions.lkj_cholesky",
    # "torch.distributions.gumbel",
    # "torch.distributions.poisson",
    # "torch.distributions.half_cauchy",
    # "torch.distributions.relaxed_bernoulli",
    # "torch.distributions.normal",
    # "torch.distributions.inverse_gamma",
    # "torch.distributions.multivariate_normal",
    # "torch.distributions.exp_family",
]

debugger = bencherDebugger(
    {
        "seed": 1234567890,
        "devices": ["cpu"],
        "test_modules": modules,
        "format": "json",
        "num_epoches": 1,
        "including_success": False
    }
)
result = debugger.run()
print("Done")
