import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.DataParallel)
class TorchNnDataparallelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_data_parallel_correctness(self):
        # Randomly generate the number of GPUs to use (between 1 and 4)
        num_gpus = 1
        device_ids = list(range(num_gpus))

        # Randomly generate the batch size (must be larger than the number of GPUs)
        batch_size = random.randint(num_gpus + 1, 10)

        # Randomly generate the number of features for the input tensor
        num_features = random.randint(1, 10)

        # Create a simple linear model
        model = torch.nn.Linear(num_features, num_features)

        # Wrap the model with DataParallel
        model_parallel = torch.nn.DataParallel(model, device_ids=device_ids)

        # Generate a random input tensor with the batch size and number of features
        input_tensor = torch.randn(batch_size, num_features)

        # Move the input tensor to the first device
        input_tensor = input_tensor.to(device_ids[0])

        # Forward pass through the DataParallel model
        output = model_parallel(input_tensor)

        return output
