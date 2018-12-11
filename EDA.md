# EDA
## Contents
[0.Import Libraries](#cdrsb1)<br>
[1.Load, Preprocess, Merge, and Split data](#cdrsb1)<br>
[2.Perform EDA to select potential predictors](#cdrsb1)<br>
    [a.)Demographics Characteristics](#cdrsb1)<br>
    [b.)Outcome: Baseline Diagnosis of Alzheimer's Disease](#cdrsb1)<br>
    [c.)Lifestyle factors (from medical history dataset)](#cdrsb1)<br>
    [d.)Neurocognitive/neuropsychological assessments](#cdrsb1)<br>
    [e.)Cerebrospinal fluid (CSF) Biomarkers](#cdrsb1)<br>
    [f.)Imaging factors](#cdrsb1)<br>
    [g.) Genetic factors](#cdrsb1)<br>
</ul>


## <a name="Import libraries"></a> 0. Import libraries

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_context('poster')

# set color palette
dpal = sns.choose_colorbrewer_palette(data_type='diverging', as_cmap=True)
```
```py
# 19 Predictors we choose based on EDA
predictors = ['AGE','gender','married','MH16SMOK','MMSE_bl','RAVLT_learning_bl',
                     'RAVLT_immediate_bl','RAVLT_perc_forgetting_bl','AVLT_Delay_Rec','ADAS13_bl',
                    'TMT_PtB_Complete','CDRSB_bl','ABETA_bl_n','TAU_bl_n','Hippocampus_bl','Entorhinal_bl',
                    'Ventricles_bl','MidTemp_bl','APOE4']
```
