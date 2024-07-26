import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.add_custom_scalars)
class TorchUtilsTensorboardWriterAddcustomscalarsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_custom_scalars_correctness(self):
        writer = SummaryWriter()

        # Randomly generate scalar names and their values
        scalar_name1 = f"scalar_{random.randint(1, 100)}"
        scalar_name2 = f"scalar_{random.randint(1, 100)}"
        scalar_value1 = random.uniform(0.1, 10.0)
        scalar_value2 = random.uniform(0.1, 10.0)

        # Randomly generate layout for custom scalars
        layout = {
            'custom_scalars': {
                'scalars': [scalar_name1, scalar_name2],
                'layout': {
                    'row1': {
                        'column1': ['custom_scalars/scalars']
                    }
                }
            }
        }

        # Add custom scalars to the writer
        writer.add_custom_scalars(layout)

        # Add scalar values to the writer
        writer.add_scalar(scalar_name1, scalar_value1)
        writer.add_scalar(scalar_name2, scalar_value2)

        # Close the writer
        writer.close()

        return layout
