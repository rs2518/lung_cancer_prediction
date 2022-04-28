import os

from .utils import (ROOT,
                    CLINICAL_INFO_COLS,
                    HISTOLOGY_MAP,
                    STAGE_MAP,
                    SMOKING_MAP)

from .utils import create_directory



create_directory(os.path.join(ROOT, "figures"))