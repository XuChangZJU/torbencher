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
from .add_video import TorchUtilsTensorboardWriterAddvideoTestCase
from .add_hparams import TorchUtilsTensorboardWriterAddhparamsTestCase
from .add_image import TorchUtilsTensorboardWriterAddimageTestCase
from .add_pr_curve import TorchUtilsTensorboardWriterAddprcurveTestCase
from .flush import TorchUtilsTensorboardWriterFlushTestCase
from .add_custom_scalars import TorchUtilsTensorboardWriterAddcustomscalarsTestCase
from .add_mesh import TorchUtilsTensorboardWriterAddmeshTestCase
from .add_images import TorchUtilsTensorboardWriterAddimagesTestCase
from .add_embedding import TorchUtilsTensorboardWriterAddembeddingTestCase
from .add_scalar import TorchUtilsTensorboardWriterAddscalarTestCase
from .add_text import TorchUtilsTensorboardWriterAddtextTestCase
from .add_graph import TorchUtilsTensorboardWriterAddgraphTestCase
from .close import TorchUtilsTensorboardWriterCloseTestCase
from .add_figure import TorchUtilsTensorboardWriterAddfigureTestCase
from .add_scalars import TorchUtilsTensorboardWriterAddscalarsTestCase
from .add_histogram import TorchUtilsTensorboardWriterAddhistogramTestCase
