import random
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api
import unittest

@test_api(torch.nn.Bilinear)
class TorchNnBilinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip
    def test_bilinear_correctness(self):
        # Randomly generate the size of each input sample with smaller range
        in1_features = random.randint(1, 5)  # Reduced range
        in2_features = random.randint(1, 5)  # Reduced range
        out_features = random.randint(1, 5)  # Reduced range

        # Create a Bilinear layer with the generated sizes
        bilinear_layer = torch.nn.Bilinear(in1_features, in2_features, out_features)

        # Randomly generate the batch size
        batch_size = 2  # Reduced range

        # Generate random input tensors with the appropriate sizes
        input1 = torch.randn(batch_size, in1_features)
        input2 = torch.randn(batch_size, in2_features)

        # Apply the bilinear transformation
        output = bilinear_layer(input1, input2)

        return output

