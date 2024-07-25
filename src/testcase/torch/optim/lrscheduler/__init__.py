# from .CyclicLR import TorchOptimLrschedulerCycliclrTestCase
# from .LinearLR import TorchOptimLrschedulerLinearlrTestCase
# from .LambdaLR import TorchOptimLrschedulerLambdalrTestCase
# from .StepLR import TorchOptimLrschedulerSteplrTestCase
# from .SequentialLR import TorchOptimLrschedulerSequentiallrTestCase
# from .CosineAnnealingLR import TorchOptimLrschedulerCosineannealinglrTestCase
# from .MultiplicativeLR import TorchOptimLrschedulerMultiplicativelrTestCase
# from .ExponentialLR import TorchOptimLrschedulerExponentiallrTestCase
# from .CosineAnnealingWarmRestarts import TorchOptimLrschedulerCosineannealingwarmrestartsTestCase
# from .ConstantLR import TorchOptimLrschedulerConstantlrTestCase
# from .OneCycleLR import TorchOptimLrschedulerOnecyclelrTestCase
# from .MultiStepLR import TorchOptimLrschedulerMultisteplrTestCase
# from .ReduceLROnPlateau import TorchOptimLrschedulerReducelronplateauTestCase
# from .ChainedScheduler import TorchOptimLrschedulerChainedschedulerTestCase
# from .PolynomialLR import TorchOptimLrschedulerPolynomiallrTestCase

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
