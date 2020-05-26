import logging
import re

# Package imports
from .constants import STATUS_FAIL, STATUS_OK  # syntatic sugar
from ._version import get_versions

__all__ = ["__version__",
           "STATUS_FAIL",
           "STATUS_OK"]

logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] %(levelname)s - %(message)s"
    )

# Create custom logger
logger = logging.getLogger(__name__)

# Define version
__version__ = re.findall(r"\d+\.\d+\.\d+", get_versions()['version'])[0]
del get_versions