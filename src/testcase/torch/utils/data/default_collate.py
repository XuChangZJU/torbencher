import random
from collections import namedtuple

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.data.default_collate)
class TorchUtilsDataDefaultUcollateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_default_collate_tensor(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        consistent_size = torch.Size(input_size)
        # 确保所有张量的尺寸一致
        batch = [torch.randn(consistent_size) for _ in range(5)]
        result = torch.utils.data.default_collate(batch)
        return result

    def test_default_collate_numpy_arrays(self):
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        consistent_size = input_size
        # 使用 .cpu() 将张量移回到 CPU 上
        batch = [torch.randn(consistent_size).cpu().numpy() for _ in range(random.randint(1, 10))]
        result = torch.utils.data.default_collate(batch)
        return result

    def test_default_collate_float(self):
        batch = [random.uniform(0.1, 10.0) for _ in range(random.randint(1, 10))]
        result = torch.utils.data.default_collate(batch)
        return result

    def test_default_collate_int(self):
        batch = [random.randint(1, 100) for _ in range(random.randint(1, 10))]
        result = torch.utils.data.default_collate(batch)
        return result

    def test_default_collate_mapping(self):
        batch = [{'A': random.randint(0, 100), 'B': random.randint(0, 100)} for _ in range(random.randint(1, 10))]
        result = torch.utils.data.default_collate(batch)
        return result

    def test_default_collate_namedtuple(self):
        Point = namedtuple('Point', ['x', 'y'])
        batch = [Point(random.randint(0, 100), random.randint(0, 100)) for _ in range(random.randint(1, 10))]
        result = torch.utils.data.default_collate(batch)
        return result

    def test_default_collate_tuple(self):
        batch = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(random.randint(1, 10))]
        result = torch.utils.data.default_collate(batch)
        return result

    def test_default_collate_list(self):
        batch = [[random.randint(0, 100), random.randint(0, 100)] for _ in range(random.randint(1, 10))]
        result = torch.utils.data.default_collate(batch)
        return result
