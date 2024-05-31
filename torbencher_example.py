from torbencher import torbencher

bencher = torbencher(
    {
        "seed": 1234567890,
        "devices": ["cuda"],
        "test_modules": [
            "torch",
            "torch.nn",
            "torch.nn.functional",
            # ...
        ],
        "format": "json",
    }
)
result = bencher.run()
print(result)
