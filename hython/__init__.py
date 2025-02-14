import logging
import urllib3 
import warnings

# Suppress multiple specific warnings
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

__version__ = "0.1.1"

logger = logging.getLogger(__name__)