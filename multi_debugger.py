import multiprocessing


def run_debugger(modules: list, result_file: str):
    from src import bencherDebugger
    debugger = bencherDebugger(
        {
            "seed": 1234567890,
            "devices": ["cpu"],
            "test_modules": modules,
            "format": "json",
            "num_epoches": 5,
            "including_success": False,
            "result_file": result_file
        }
    )
    result = debugger.run()


if __name__ == '__main__':
    process1 = multiprocessing.Process(target=run_debugger, args=[["torch"], "torch_result.csv"])
    process2 = multiprocessing.Process(target=run_debugger, args=[["torch.linalg"], "linalg_result.csv"])
    process1.start()
    process2.start()
    process1.join()
    process2.join()
