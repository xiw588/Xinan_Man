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
## <a name="Load, Preprocess, Merge, and Split data"></a> 1. Load, Preprocess, Merge, and Split data
```py
# KEY ADNI table with age, gender, ethnicity, race, education, marital status, and APOE status
# ADNIMERGE data contains part of biomarker data, part of Neuropsychological data, and key feature in FAQ dataset
adnimerge=pd.read_csv('ADNIMERGE.csv',low_memory=False)
# Create a binary variable for maritial status and for gender
adnimerge['PTMARRY'] = np.where(adnimerge['PTMARRY']=='Married','Married','Unmarried')
adnimerge_bl= adnimerge[adnimerge['VISCODE'] == 'bl']

# Biomarker 
biomarker = pd.read_csv('UPENNBIOMK_MASTER.csv')
biomarker_bl = biomarker.loc[(biomarker['VISCODE'] == 'bl') & (biomarker['BATCH'] == 'MEDIAN')]

# Medical History table with smoking and alcohol abuse variables
mh = pd.read_csv('MEDHIST.csv') 
mh = mh[mh.duplicated(subset='RID',keep='first')==False]

# Neuropsychological
item = pd.read_csv('ITEM.csv')
item_sub = item.loc[:,['RID','VISCODE','TMT_PtA_Complete','TMT_PtB_Complete','AVLT_Delay_Rec']]
item_bl = item_sub.loc[(item_sub['VISCODE'] == 'bl')]

# merge datasets
data = pd.merge(adnimerge_bl, biomarker_bl, how='inner', on='RID')
data = pd.merge(data, mh, how='inner', on=['RID'])  
data = pd.merge(data, item_bl, on='RID', how='inner')

# Split data into training and test set - All the EDA are performed on the training set
np.random.seed(9001)
msk = np.random.rand(len(data)) < 0.75
data_train = data[msk]
data_test = data[~msk]

print('The sample size of full dataset is %.d.' % data.shape[0])
print('The sample size of training set is %.d (%.2f' % (data_train.shape[0], 
                                                        (100*data_train.shape[0]/data.shape[0])),'%).')
print('The sample size of test set is %.d (%.2f' % (data_test.shape[0], 
                                                        (100*data_test.shape[0]/data.shape[0])),'%).')
                                                        
data_train.to_csv('data_train.csv')
data_test.to_csv('data_test.csv')
```
## <a name="Perform EDA to select potential predictors"></a> 2. Perform EDA to select potential predictors
## <a name="a. Demographics Characteristics"></a>a. Demographics Characteristics
## <a name="Outcome: Baseline Diagnosis of Alzheimer's Disease"></a> Outcome: Baseline Diagnosis of Alzheimer's Disease

```py
bldx_df=pd.DataFrame(index=['Baseline Diagnosis Prevalence'],columns=['CN','AD','LMCI'])
bldx_df.CN=np.mean(data_train['DX_bl']=='CN')
bldx_df.AD=np.mean(data_train['DX_bl']=='AD')
bldx_df.LMCI=np.mean(data_train['DX_bl']=='LMCI')
bldx_df
```	
                                CN	        AD          LMCI
                                
Baseline Diagnosis Prevalence	0.294964	0.230216	0.47482


