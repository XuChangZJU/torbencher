import multiprocessing

public_config = {
    "out_dir": "./results",
    "seed": 1234567890,
    "devices": [
        "cpu",
        "cuda",
    ],
    "test_modules": [
        "torch",
        "torch.Tensor",
        "torch.nn",
        "torch.nn.functional",
        "torch.special",
        "torch.autograd",
        "torch.utils.checkpoint",
        "torch.export",
        "torch.profiler",
        "torch.profiler.itt",
        "torch.testing",
        "torch.nn.init",
        "torch.fft",
        "torch.nn.parallel",
        "torch.nn.utils",
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
        "torch.optim",
        "torch.optim.Optimizer",
        "torch.optim.lrscheduler",
        "torch.utils.data",
        "torch.utils.data._utils",
        "torch.utils.data.distributed",
        "torch.utils.data._utils.collate",
    ],
    "format": "csv",
    "num_epoch": 1,
    "name_spec": "timestamp",
    "debug": False
}


def task(task_config: dict):
    from src.torbencherc import torbencherc
    bencher = torbencherc(task_config)
    result = bencher.run()
    print(result)
    return result


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    task_results = []
    for i in range(len(public_config["test_modules"])):
        task_config = public_config.copy()
        task_config["test_modules"] = [
            task_config["test_modules"][i]
        ]
        task_results.append(pool.apply_async(task, (task_config,)))
    pool.close()
    pool.join()
