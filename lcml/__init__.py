import os

from .utils import (ROOT,
                    CLINICAL_INFO_COLS,
                    STATUS_MAP,
                    HISTOLOGY_MAP,
                    STAGE_MAP,
                    SMOKING_MAP,
                    SAMPLE_IDS)

from .utils import create_directory



create_directory(os.path.join(ROOT, "figures"))