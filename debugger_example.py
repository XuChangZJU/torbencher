from src import bencherDebugger

modules = [
    "torch.export",
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
