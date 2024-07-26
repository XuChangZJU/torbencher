import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.hub.help)
class TorchHubHelpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hub_help_correctness(self):
        # Randomly generate repo_owner and repo_name
        repo_owner = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(3, 10)))
        repo_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(3, 10)))

        # Randomly decide whether to include a ref
        if random.choice([True, False]):
            ref = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=random.randint(3, 10)))
            github = f"{repo_owner}/{repo_name}:{ref}"
        else:
            github = f"{repo_owner}/{repo_name}"

        # Randomly generate model name
        model = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(3, 10)))

        # Call torch.hub.help with generated parameters
        result = torch.hub.help(github, model)

        return result
