import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.generate_methods_for_privateuse1_backend)
class TorchUtilsGenerateUmethodsUforUprivateuse1UbackendTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_generate_methods_for_privateuse1_backend_correctness(self):
        # Randomly decide whether to register methods for torch.Tensor, torch.nn.Module, and torch.Storage
        for_tensor = random.choice([True, False])
        for_module = random.choice([True, False])
        for_storage = random.choice([True, False])

        # Randomly generate a list of unsupported dtypes for storage methods
        all_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8]
        unsupported_dtype = random.sample(all_dtypes, random.randint(0, len(all_dtypes)))

        # Rename the privateuse1 backend to a random name
        backend_name = "backend_" + str(random.randint(1, 1000))
        
        torch.utils.rename_privateuse1_backend(backend_name)

        # Generate methods for the renamed backend
        torch.utils.generate_methods_for_privateuse1_backend(for_tensor, for_module, for_storage, unsupported_dtype)

        # Test if the methods are correctly generated for torch.Tensor
        if for_tensor:
            tensor = torch.randn(3, 3)
            result_tensor_method = getattr(tensor, backend_name)()
            result_tensor_attr = getattr(tensor, f"is_{backend_name}")

        # Test if the methods are correctly generated for torch.nn.Module
        if for_module:
            module = torch.nn.Linear(3, 3)
            result_module_method = getattr(module, backend_name)()
            result_module_attr = getattr(module, f"is_{backend_name}")

        # Test if the methods are correctly generated for torch.Storage
        if for_storage:
            storage = torch.FloatStorage(3)
            result_storage_method = getattr(storage, backend_name)()
            result_storage_attr = getattr(storage, f"is_{backend_name}")

        return (for_tensor, for_module, for_storage, unsupported_dtype)
