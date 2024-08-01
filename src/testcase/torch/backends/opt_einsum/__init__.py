# from .get_opt_einsum import TorchBackendsOpteinsumGetopteinsumTestCase
# from .is_available import TorchBackendsOpteinsumIsavailableTestCase

import os
import importlib
import logging
from inspect import isclass

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
    try:
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isclass(attribute) and issubclass(attribute, TorBencherTestCaseBase)\
                    and attribute is not TorBencherTestCaseBase:
                globals()[attribute_name] = attribute
    except Exception as e:
        raise ValueError(f"The testcase that cause error is {attribute_name}") from e
