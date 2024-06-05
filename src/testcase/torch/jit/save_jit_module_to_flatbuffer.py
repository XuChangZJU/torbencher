
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.save_jit_module_to_flatbuffer)
class TorchJitSaveJitModuleToFlatbufferTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_save_jit_module_to_flatbuffer_correctness(self):
        class TestModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(random.randint(1, 10), random.randint(1, 10))

            @torch.jit.script_method
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        script_module = TestModule()
        result = torch.jit.save_jit_module_to_flatbuffer(script_module, f"test_model_{random.randint(1, 10)}.pt")
        return result

    @test_api_version.larger_than("1.1.3")
    def test_save_jit_module_to_flatbuffer_large_scale(self):
        class TestModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(random.randint(1000, 10000), random.randint(1000, 10000))

            @torch.jit.script_method
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        script_module = TestModule()
        result = torch.jit.save_jit_module_to_flatbuffer(script_module, f"test_model_{random.randint(1, 10)}.pt")
        return result

