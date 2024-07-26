import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.__init__)
class TorchUtilsTensorboardWriterInitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tensorboard_writer_init_correctness(self):
        log_dir = f"runs/test_{random.randint(1, 1000)}"  # Random log directory
        comment = f"test_{random.randint(1, 1000)}"  # Random comment
        purge_step = random.randint(0, 10)  # Random purge step
        max_queue = random.randint(1, 10)  # Random max queue size
        flush_secs = random.uniform(1.0, 10.0)  # Random flush seconds
        filename_suffix = f"_{random.randint(1, 1000)}"  # Random filename suffix

        writer = SummaryWriter(log_dir=log_dir, comment=comment, purge_step=purge_step,
                               max_queue=max_queue, flush_secs=flush_secs,
                               filename_suffix=filename_suffix)
        return writer

    from .SummaryWriter import TorchUtilsTensorboardWriterSummarywriterTestCase


# from .add_video import TorchUtilsTensorboardWriterAddvideoTestCase
# from .add_hparams import TorchUtilsTensorboardWriterAddhparamsTestCase
# from .add_image import TorchUtilsTensorboardWriterAddimageTestCase
# from .add_pr_curve import TorchUtilsTensorboardWriterAddprcurveTestCase
# from .flush import TorchUtilsTensorboardWriterFlushTestCase
# from .add_custom_scalars import TorchUtilsTensorboardWriterAddcustomscalarsTestCase
# from .add_mesh import TorchUtilsTensorboardWriterAddmeshTestCase
# from .add_images import TorchUtilsTensorboardWriterAddimagesTestCase
# from .add_embedding import TorchUtilsTensorboardWriterAddembeddingTestCase
# from .add_scalar import TorchUtilsTensorboardWriterAddscalarTestCase
# from .add_text import TorchUtilsTensorboardWriterAddtextTestCase
# from .add_graph import TorchUtilsTensorboardWriterAddgraphTestCase
# from .close import TorchUtilsTensorboardWriterCloseTestCase
# from .add_figure import TorchUtilsTensorboardWriterAddfigureTestCase
# from .add_scalars import TorchUtilsTensorboardWriterAddscalarsTestCase
# from .add_histogram import TorchUtilsTensorboardWriterAddhistogramTestCase

import os
import importlib
import logging

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase

# 设置日志配置
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_directory: str = os.path.dirname(os.path.abspath(__file__))
script_files: list = [f for f in os.listdir(current_directory) if f.endswith('.py') and f != '__init__.py']

for script_file in script_files:
    module_name: str = script_file[:-3]  # Remove the .py extension
    try:
        module = importlib.import_module(f'.{module_name}', package=__package__)
        # logger.debug(f"Successfully imported module {module_name}")
    except Exception as e:
        logger.debug(f"Failed to import module {module_name}: {e}")
        continue

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isinstance(attribute, type) and issubclass(attribute, TorBencherTestCaseBase):
            globals()[attribute_name] = attribute
