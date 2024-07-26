import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.tensorboard.writer.add_text)
class TorchUtilsTensorboardWriterAddtextTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_text_correctness(self):
        writer = SummaryWriter()
        
        # Random tag for the text
        tag = f"test_tag_{random.randint(1, 100)}"
        
        # Random text content
        text_string = f"Random text content {random.randint(1, 100)}"
        
        # Random global step
        global_step = random.randint(1, 100)
        
        # Adding text to the writer
        writer.add_text(tag, text_string, global_step)
        
        # Closing the writer
        writer.close()
        
        return f"Text added with tag: {tag}, content: {text_string}, at step: {global_step}"
    