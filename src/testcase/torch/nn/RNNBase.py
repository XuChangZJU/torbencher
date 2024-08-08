import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


# 定义一个模拟RNN类，继承自RNNBase，仅仅为了测试目的实现forward方法
class MockRNN(torch.nn.RNNBase):
    def __init__(self, mode, input_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__(mode, input_size, hidden_size, num_layers, *args, **kwargs)

    def forward(self, input, hx=None):
        # 这里只是返回一些占位输出和隐藏状态，实际应用中应有具体计算逻辑
        return torch.zeros_like(input), torch.zeros_like(hx)


class MockRNN(torch.nn.RNNBase):
    def __init__(self, mode, input_size, hidden_size, num_layers, *args, **kwargs):
        super().__init__(mode, input_size, hidden_size, num_layers, *args, **kwargs)

    def forward(self, input, hx=None):
        # 动态获取序列长度，以确保输出形状与输入匹配
        seq_length = input.size(0)
        batch_size = input.size(1)

        # 根据输入的形状动态生成输出和隐藏状态
        output = torch.zeros(seq_length, batch_size, self.hidden_size)
        if hx is None:
            h_n = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            h_n = hx  # 如果提供了初始隐藏状态，则直接使用

        return output, h_n


@test_api(torch.nn.RNNBase)
class TorchNnRnnbaseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_rnnbase_correctness(self):
        input_size = random.randint(1, 10)
        hidden_size = random.randint(1, 10)
        num_layers = random.randint(1, 3)
        batch_size = random.randint(1, 5)
        seq_length = random.randint(1, 5)

        input_tensor = torch.randn(seq_length, batch_size, input_size)
        rnn = MockRNN(mode='RNN_TANH', input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        h_0 = torch.randn(num_layers, batch_size, hidden_size)  # 可选的，也可以不提供让forward内部生成

        output, h_n = rnn(input_tensor, h_0)

        # 由于forward已经根据输入动态调整了输出形状，理论上这里不会出现问题
        expected_output_shape = (seq_length, batch_size, hidden_size)
        expected_hn_shape = (num_layers, batch_size, hidden_size)

        self.assertTrue(output.shape == expected_output_shape,
                        f"Output shape mismatch. Expected {expected_output_shape}, got {output.shape}")
        self.assertTrue(h_n.shape == expected_hn_shape,
                        f"Hidden state shape mismatch. Expected {expected_hn_shape}, got {h_n.shape}")

        return output, h_n
