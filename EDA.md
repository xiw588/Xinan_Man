# EDA
## Contents
[0.Import Libraries](#cdrsb1)<br>
[1.Load, Preprocess, Merge, and Split data](#cdrsb1)<br>
[2.Perform EDA to select potential predictors](#cdrsb1)<br>
    [a.)Demographics Characteristics](#cdrsb1)<br>
    [b.)Outcome: Baseline Diagnosis of Alzheimer's Disease](#cdrsb1)<br>
    [c.)Lifestyle factors (from medical history dataset)](#cdrsb1)<br>
    [d.)Neurocognitive/neuropsychological assessments](#cdrsb1)<br>
      [1.)Histogram](#cdrsb1)<br>
      [2.)Box plots and count plots with Outcomes](#cdrsb1)<br>
      [3.)Correlation matrix](#cdrsb1)<br>
    [e.)Cerebrospinal fluid (CSF) Biomarkers](#cdrsb1)<br>
      [1.)Histogram](#cdrsb1)<br>
      [2.)Box plots](#cdrsb1)<br>
      [3.)Correlation matrix](#cdrsb1)<br>
    [f.)Imaging factors](#cdrsb1)<br>
      [1.)Histogram](#cdrsb1)<br>
      [2.)Box plots](#cdrsb1)<br>
      [3.)Correlation matrix](#cdrsb1)<br>
    [g.) Genetic factors](#cdrsb1)<br>
</ul>


## <a name="data-preparation"></a> 1. Data Preparation
#### <a name="reading-and-cleaning-data"></a> 1) Reading and Cleaning Data

```py
# import dataset
data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')

# functions to recode some variable
def change_gender(x):    # recode gender
    if x == 'Female':
        return 1
    else:
        return 0

def change_marry(x):     # recode marry
    if x == 'Married':
        return 1
    else:
        return 0

def change_abeta_bl(x):   # recode ABETA_bl
    if x == '>1700':
        return 1800
    elif x == '<200':
        return 100
    else:
        return float(x)

def change_tau_bl(x):      # recode TAU_bl
    if x == '<80':
        return 40
    else:
        return float(x)

def change_dx_bl(x):       # recode DX_bl(outcome)
    if x == 'CN':
        return 1
    elif x == 'AD':
        return 2
    else:
        return 3

def process_data(df):
    df['gender'] = df.PTGENDER.apply(change_gender)
    df['married'] = df.PTMARRY.apply(change_marry)
    df['ABETA_bl_n'] = df.ABETA_bl.apply(change_abeta_bl)
    df['TAU_bl_n'] = df.TAU_bl.apply(change_tau_bl)
    df['y'] = df.DX_bl.apply(change_dx_bl)
    return df
```
```py
# 19 Predictors we choose based on EDA
predictors = ['AGE','gender','married','MH16SMOK','MMSE_bl','RAVLT_learning_bl',
                     'RAVLT_immediate_bl','RAVLT_perc_forgetting_bl','AVLT_Delay_Rec','ADAS13_bl',
                    'TMT_PtB_Complete','CDRSB_bl','ABETA_bl_n','TAU_bl_n','Hippocampus_bl','Entorhinal_bl',
                    'Ventricles_bl','MidTemp_bl','APOE4']
```
