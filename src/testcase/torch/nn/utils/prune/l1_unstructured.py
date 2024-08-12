import random

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.utils.prune.l1_unstructured)
class TorchNnUtilsPruneL1UunstructuredTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_l1_unstructured_correctness(self):
        dim = random.randint(1, 4)  # 随机维度数
        num_of_elements_each_dim = random.randint(1, 5)  # 每个维度的随机元素数量
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        output_size = random.randint(1, 5)  # 输出大小

        # 创建一个线性层
        module = nn.Linear(num_of_elements_each_dim, output_size)

        # 随机生成输入数据
        input_data = torch.randn(input_size)

        # 随机剪枝比例
        amount = random.uniform(0.0, 1.0) if random.choice([True, False]) else random.randint(1, module.weight.numel())

        # 应用 L1 Unstructured 剪枝
        prune.l1_unstructured(module, name='weight', amount=amount)

        # 这里你可以检查剪枝后的权重，或者前向传播输入数据以验证结果
        pruned_output = module(input_data)

        return pruned_output
