from src.torbencherc import torbencherc

config = {
    "out_dir": "./results",
    "seed": 1234567890,
    "devices": [
        "cuda",
        # other device names...
    ],
    "test_modules": [
        # "torch",
        # "torch.nn",
        "torch.nn.functional",
        # other torch package names...
    ],
    "format": "json",
    "num_epoch": 3
}

bencher = torbencherc(config)
result = bencher.run()
print(result)
