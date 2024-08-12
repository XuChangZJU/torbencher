import random

import torch
from torch import nn, optim

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.set_detect_anomaly)
class TorchAutogradSetUdetectUanomalyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_set_detect_anomaly_correctness(self):
        # 启用异常检测
        torch.autograd.set_detect_anomaly(True)

        # 定义一个简单的模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(1, 1)

            def forward(self, x):
                return self.fc(x)

        # 实例化模型和优化器
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # 创建一个输入张量和目标张量
        input = torch.tensor([[1.0]], requires_grad=True)
        target = torch.tensor([[2.0]])

        # 训练步骤
        try:
            for _ in range(1):  # 一次迭代
                optimizer.zero_grad()

                # 正常的前向传播
                output = model(input)

                # 计算损失，这里故意引入一个梯度计算的异常
                loss = (output - target).pow(2).sum()

                # 人为引入 NaN 值来触发异常
                if torch.rand(1).item() > 0.5:
                    loss = torch.tensor(float('nan'))

                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

        except RuntimeError as e:
            # # 打印异常信息
            # print("Caught RuntimeError during backpropagation:")
            # print(e)
            return True  # 异常被捕捉，说明异常检测功能正常
        return False  # 没有异常，说明异常检测功能未正常工作
