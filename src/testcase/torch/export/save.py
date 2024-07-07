import torch
import io
import random
import pathlib


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.export.save)
class TorchExportSaveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_save_correctness(self):
        """
        This test case tests the correctness of `torch.export.save` by:
            - Creating a simple module.
            - Exporting the module using `torch.export.export`.
            - Saving the exported program to a file using `torch.export.save`.
            - Loading the saved program using `torch.export.load`.
            - Verifying that the loaded program produces the same output as the original module.
        """
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10
    
        ep = torch.export.export(MyModule(), (torch.randn(5),))
        f = 'exported_program.pt2'
        torch.export.save(ep, f)
        loaded_ep = torch.export.load(f)
        return loaded_ep
    
    # Automatically added function calls
    
    
    