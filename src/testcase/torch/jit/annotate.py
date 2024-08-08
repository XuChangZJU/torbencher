import random
from typing import Dict

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.jit.annotate)
class TorchJitAnnotateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_annotate_correctness(self):
        # Randomly generate type and value for annotation
        random_int = random.randint(0, 100)
        random_float = random.uniform(0, 100)
        random_str = "test_string"
        random_tensor = torch.randn([random.randint(1, 10) for _ in range(random.randint(1, 4))])
        random_list = [random.randint(0, 100) for _ in range(random.randint(1, 10))]
        random_dict = {str(i): random.randint(0, 100) for i in range(random.randint(1, 10))}

        # Use torch.jit.annotate to annotate different types
        annotated_int = torch.jit.annotate(int, random_int)
        annotated_float = torch.jit.annotate(float, random_float)
        annotated_str = torch.jit.annotate(str, random_str)
        annotated_tensor = torch.jit.annotate(torch.Tensor, random_tensor)
        annotated_list = torch.jit.annotate(list, random_list)
        annotated_dict = torch.jit.annotate(Dict[str, int], random_dict)

        # Return a dictionary containing all annotated values
        return {"annotated_int": annotated_int,
                "annotated_float": annotated_float,
                "annotated_str": annotated_str,
                "annotated_tensor": annotated_tensor,
                "annotated_list": annotated_list,
                "annotated_dict": annotated_dict}
