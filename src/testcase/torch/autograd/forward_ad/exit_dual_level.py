import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.forward_ad.exit_dual_level)
class TorchAutogradForwardUadExitUdualUlevelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_exit_dual_level_correctness(self):
        # 创建一个随机张量
        dim = random.randint(1, 4)  # 随机维度
        num_of_elements_each_dim = random.randint(1, 5)  # 每个维度的元素数量
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        tensor = torch.randn(input_size, requires_grad=True)

        # 进入前向微分层级
        with torch.autograd.forward_ad.dual_level():
            # 在前向微分层级内执行一些操作
            tensor_squared = tensor * tensor

        # 在层级外执行另一个操作，检查层级是否已退出
        tensor_cubed = tensor * tensor * tensor

        return tensor_squared, tensor_cubed
