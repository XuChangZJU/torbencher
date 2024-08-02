import torch
import random
from torch.utils.checkpoint import checkpoint_sequential

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version


class TorchUtilsCheckpointCheckpointsequentialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_checkpoint_sequential_correctness(self):
        # 随机生成层的数量
        num_layers = random.randint(2, 5)

        # 创建一个包含随机线性层的列表，并手动初始化参数
        layers = []
        input_size = random.randint(1, 10)
        for _ in range(num_layers):
            output_size = random.randint(1, 10)
            layer = torch.nn.Linear(input_size, output_size)
            layer.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=layer.weight.shape))
            layer.bias = torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=layer.bias.shape))
            layers.append(layer)
            input_size = output_size

        # 随机生成段的数量
        num_segments = random.randint(1, num_layers)

        # 随机输入张量
        batch_size = random.randint(1, 10)
        input_features = layers[0].in_features
        input_tensor = torch.randn(batch_size, input_features)

        # 应用 checkpoint_sequential
        result = checkpoint_sequential(layers, num_segments, input_tensor)
        return result
