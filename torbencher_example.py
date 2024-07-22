from src import torbencher

bencher = torbencher(
    {
        "seed": 1234567890,
        "devices": [
            "cuda",
            # other device names...
        ],
        "test_modules": [
            "torch",
            "torch.nn",
            "torch.nn.functional",
            # other torch package names...
        ],
        "format": "json",
    }
)
result = bencher.run()
print(result)
