import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.StackDataset)
class TorchUtilsDataStackdatasetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_stackdataset_correctness(self):
        num_datasets = random.randint(1, 5)  # Random number of datasets to stack
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        datasets = []
        for _ in range(num_datasets):
            datasets.append(torch.utils.data.TensorDataset(torch.randn(input_size)))
        tuple_stack = torch.utils.data.StackDataset(*datasets)
        dict_stack = torch.utils.data.StackDataset(**{str(i): dataset for i, dataset in enumerate(datasets)})
        index = random.randint(0, len(datasets[0]) - 1)  # Random valid index
        tuple_result = tuple_stack[index]
        dict_result = dict_stack[index]
        return tuple_result, dict_result
    
    
    
    