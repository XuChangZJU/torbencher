import random
from typing import Dict

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.jit.Attribute)
class TorchJitAttributeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_jit_attribute_correctness(self):
        class TestModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # Random float value for foo
                foo_value = random.uniform(0.1, 10.0)
                self.foo = torch.jit.Attribute(foo_value, float)
                assert 0.0 < self.foo

                # Random dictionary for names_ages
                names_ages_value = {f"name_{i}": random.randint(1, 100) for i in range(random.randint(1, 5))}
                self.names_ages = torch.jit.Attribute(names_ages_value, Dict[str, int])
                for key in self.names_ages:
                    assert isinstance(self.names_ages[key], int)

        test_module = TestModule()
        return test_module.foo, test_module.names_ages
