import osROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))CLINICAL_INFO_COLS = ["Dataset", "Disease Status", "Gender", "Race",                      "Age", "Stage", "Histology", "Survival Month",                      "Survival", "Smoking", "TNM stage (T)", "TNM stage (N)",                      "TNM stage (M)", "Recurrence", "Others"]HISTOLOGY_MAP = {"ADC":"ADC",                 "Healthy":"Healthy",                 "LCC":"LCC",                 "SCC":"SCC",                 "Adenosquamous carcinoma":"ASC",                 "Large cell Neuroendocrine carcinoma":"LCNEC",                 "NSCLC-favor adenocarcinoma":"NFA",                 "NSClarge cell carcinoma-mixed":"Mixed",                 "Other":"Other"}STAGE_MAP = {" 1A":"1A",             " 1B":"1B",             " 2A":"2A",             " 2B":"2B",             " pT2N0":"2",             "1":"1",             "1A":"1A",             "1B":"1B",             "2":"2",             "2A":"2A",             "2B":"2B",             "3A":"3A",             "3B":"3B",             "4":"4"}SMOKING_MAP = {"Current":"Current",               "Ever-smoker":"Former",               "Ex-smoker":"Former",               "Former":"Former",               "Never":"Never",               "Never-smoker":"Never",               "Non-smoking":""}