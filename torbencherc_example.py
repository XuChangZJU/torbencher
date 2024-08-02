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
    "format": "csv",
    "num_epoch": 1,
    "name_spec": "timestamp"
}

bencher = torbencherc(config)
result = bencher.run()
print(result)
