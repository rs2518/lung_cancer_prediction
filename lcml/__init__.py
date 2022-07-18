import os

from .utils import (ROOT,
                    CLINICAL_INFO_COLS,
                    STATUS_MAP,
                    HISTOLOGY_MAP,
                    STAGE_MAP,
                    SMOKING_MAP,
                    TRAIN_TEST_SPLIT,
                    DATASET_PALETTE,
                    STATUS_PALETTE,
                    GENDER_PALETTE,
                    HISTOLOGY_PALETTE,
                    SMOKING_PALETTE,
                    STAGE_PALETTE)

from .utils import (create_directory,
                    load_clinical_info,
                    load_patient_status,
                    load_expression_data,
                    load_de_analysis,
                    load_de_rank,
                    load_data,
                    get_de_genes)



create_directory(os.path.join(ROOT, "figures"))