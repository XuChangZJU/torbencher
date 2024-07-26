import torch
import random
from torch.autograd.forward_ad import unpack_dual, enter_dual_level, exit_dual_level

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.forward_ad.UnpackedDualTensor)
class TorchAutogradForwardadUnpackeddualtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unpack_dual_correctness(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        primal = torch.randn(input_size)
        tangent = torch.randn(input_size)

        # 设置双数级别
        level = enter_dual_level()
        try:
            # 在这个级别下创建双数张量
            dual_tensor = torch.autograd.forward_ad.make_dual(primal, tangent)
            unpacked_dual_tensor = unpack_dual(dual_tensor)
        finally:
            # 恢复到正常的张量操作
            exit_dual_level()

        # 返回结果或进行断言检查
        return unpacked_dual_tensor
