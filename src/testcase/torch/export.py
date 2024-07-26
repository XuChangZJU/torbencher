import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.export)
class TorchExportTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_correctness(self):
        # input_size = [random.randint(1, 3) for _ in range(2)]  # Random dimension for input tensor
        # tensor = torch.randn(input_size)
        # model = torch.nn.Linear(input_size[1], random.randint(1, 5))  # Random Linear Model with valid dimensions
        #
        # # Setting up the function to export
        # def model_to_export(x):
        #     return model(x)
        #
        # # Exporting the model
        # scripted_model = torch.jit.script(model_to_export)

        # 随机生成线性层的输入和输出尺寸
        input_dim = random.randint(50, 150)
        output_dim = random.randint(5, 20)

        class MyModule(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.lin = torch.nn.Linear(input_dim, output_dim)

            def forward(self, x, y):
                return torch.nn.functional.relu(self.lin(x + y), inplace=True)

        mod = MyModule(input_dim, output_dim)
        batch_size = random.randint(1, 10)

        x_input = torch.randn(batch_size, input_dim)
        y_input = torch.randn(batch_size, input_dim)

        # 导出模型
        exported_mod = torch.export.export(mod, (x_input, y_input))

        return exported_mod
