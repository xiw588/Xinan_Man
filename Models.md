# Models
## Contents

### 1.Data Preparation
#### 1) Reading and Cleaning Data

```Markdown
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
```Markdown
# 19 Predictors we choose based on EDA
predictors = ['AGE','gender','married','MH16SMOK','MMSE_bl','RAVLT_learning_bl',
                     'RAVLT_immediate_bl','RAVLT_perc_forgetting_bl','AVLT_Delay_Rec','ADAS13_bl',
                    'TMT_PtB_Complete','CDRSB_bl','ABETA_bl_n','TAU_bl_n','Hippocampus_bl','Entorhinal_bl',
                    'Ventricles_bl','MidTemp_bl','APOE4']
```

#### 2) Imputing and Scaling Data
**We used three imputation methods to impute missing data:**
1. Deleting all missing in predictors
2. Mean imputation
3. Regression imputation

```Markdown
# 1. no imputation: drop missing values

data_train_sub1 = data_train[['AGE','PTGENDER','PTMARRY','MH16SMOK','MMSE_bl','RAVLT_learning_bl',
                     'RAVLT_immediate_bl','RAVLT_perc_forgetting_bl','AVLT_Delay_Rec','ADAS13_bl',
                    'TMT_PtB_Complete','CDRSB_bl','ABETA_bl','TAU_bl','Hippocampus_bl','Entorhinal_bl',
                    'Ventricles_bl','MidTemp_bl','APOE4','DX_bl']]
data_test_sub1 = data_test[['AGE','PTGENDER','PTMARRY','MH16SMOK','MMSE_bl','RAVLT_learning_bl',
                     'RAVLT_immediate_bl','RAVLT_perc_forgetting_bl','AVLT_Delay_Rec','ADAS13_bl',
                    'TMT_PtB_Complete','CDRSB_bl','ABETA_bl','TAU_bl','Hippocampus_bl','Entorhinal_bl',
                    'Ventricles_bl','MidTemp_bl','APOE4','DX_bl']]
data_train_sub1 = data_train_sub1.dropna()
data_test_sub1 = data_test_sub1.dropna()
data_train_sub1 = process_data(data_train_sub1)
data_test_sub1 = process_data(data_test_sub1)

# x and y
x_train1 = data_train_sub1[predictors]
x_test1 = data_test_sub1[predictors]
y_train1 = data_train_sub1.y
y_test1 = data_test_sub1.y

# data need to be scaled
scale_col = ['AGE','MMSE_bl','RAVLT_learning_bl','RAVLT_immediate_bl','RAVLT_perc_forgetting_bl',
             'AVLT_Delay_Rec','ADAS13_bl','TMT_PtB_Complete','CDRSB_bl','ABETA_bl_n','TAU_bl_n',
             'Hippocampus_bl','Entorhinal_bl','Ventricles_bl','MidTemp_bl']

# scale the data
scaler1 = StandardScaler().fit(x_train1[scale_col])
X_train1 = x_train1.copy()
X_test1 = x_test1.copy()
X_train1[scale_col] = scaler1.transform(X_train1[scale_col])
X_test1[scale_col] = scaler1.transform(X_test1[scale_col])
```

```Markdown
# 2. mean imputation

data_full = pd.concat([data_train, data_test])
data_full = data_full[['AGE','PTGENDER','PTMARRY','MH16SMOK','MMSE_bl','RAVLT_learning_bl',
                     'RAVLT_immediate_bl','RAVLT_perc_forgetting_bl','AVLT_Delay_Rec','ADAS13_bl',
                    'TMT_PtB_Complete','CDRSB_bl','ABETA_bl','TAU_bl','Hippocampus_bl','Entorhinal_bl',
                    'Ventricles_bl','MidTemp_bl','APOE4','DX_bl']]
data_full = process_data(data_full)
data_full = data_full[predictors + ['y']]

# mean imputation
imp_mean = SimpleImputer(copy=True, strategy='mean').fit(data_full[predictors])
imp_mean_v = imp_mean.transform(data_full[predictors])
data_full_new = pd.concat([pd.DataFrame(imp_mean_v), 
                       pd.DataFrame(data_full.y).reset_index(drop=True,inplace=False)], axis=1)
data_full_new.columns = predictors + ['y']

# split train and test
np.random.seed(9001)
msk = np.random.rand(len(data_full)) < 0.75
data_train_sub2 = data_full_new[msk]
data_test_sub2 = data_full_new[~msk]

# x and y
x_train2 = data_train_sub2[predictors]
x_test2 = data_test_sub2[predictors]
y_train2 = data_train_sub2.y
y_test2 = data_test_sub2.y

# scale the data
scaler2 = StandardScaler().fit(x_train2[scale_col])
X_train2 = x_train2.copy()
X_test2 = x_test2.copy()
X_train2[scale_col] = scaler2.transform(X_train2[scale_col])
X_test2[scale_col] = scaler2.transform(X_test2[scale_col])
```

```Markdown
# 3. regression imputation

data_full = pd.concat([data_train, data_test])
data_full = data_full[['AGE','PTGENDER','PTMARRY','MH16SMOK','MMSE_bl','RAVLT_learning_bl',
                     'RAVLT_immediate_bl','RAVLT_perc_forgetting_bl','AVLT_Delay_Rec','ADAS13_bl',
                    'TMT_PtB_Complete','CDRSB_bl','ABETA_bl','TAU_bl','Hippocampus_bl','Entorhinal_bl',
                    'Ventricles_bl','MidTemp_bl','APOE4','DX_bl']]
data_full = process_data(data_full)
data_full = data_full[predictors + ['y']]

# regression for imputation
missing_col = data_full.columns[data_full.isnull().any()].tolist()

for v in missing_col:
    x_v = data_full[~data_full[v].isnull()].drop(columns = missing_col+['y'])
    y_v = data_full[~data_full[v].isnull()][v]
    imp_linear = LinearRegression().fit(x_v, y_v)
    y_hat = imp_linear.predict(x_v)
    x_missing = data_full[data_full[v].isnull()].drop(columns = missing_col+['y'])
    y_missing = imp_linear.predict(x_missing)
    y_missing_noise = y_missing + np.random.normal(loc = 0,scale = np.sqrt(\
                    metrics.mean_squared_error(y_v, y_hat)),size = y_missing.shape[0])
    missing_index = data_full[data_full[v].isnull()].index
    missing_series = pd.Series(data = y_missing_noise, index = missing_index)
    data_full[v] = data_full[v].fillna(missing_series)

# split train and test
np.random.seed(9001)
msk = np.random.rand(len(data_full)) < 0.75
data_train_sub3 = data_full[msk]
data_test_sub3 = data_full[~msk]

# x and y
x_train3 = data_train_sub3[predictors]
x_test3 = data_test_sub3[predictors]
y_train3 = data_train_sub3.y
y_test3 = data_test_sub3.y

# scale the data
scaler3 = StandardScaler().fit(x_train3[scale_col])
X_train3 = x_train3.copy()
X_test3 = x_test3.copy()
X_train3[scale_col] = scaler3.transform(X_train3[scale_col])
X_test3[scale_col] = scaler3.transform(X_test3[scale_col])
```

```Markdown
# Print the shape of each dataset
print('1. Drop all missing: \n\tThe sample size of training set : test set is {} ({:.2f}%) : {} ({:.2f}%).' 
      .format(X_train1.shape[0], (100*X_train1.shape[0]/(X_train1.shape[0]+X_test1.shape[0])),
             X_test1.shape[0], (100*X_test1.shape[0]/(X_train1.shape[0]+X_test1.shape[0]))))

print('2. Mean imputation: \n\tThe sample size of training set : test set is {} ({:.2f}%) : {} ({:.2f}%).' 
      .format(X_train2.shape[0], (100*X_train2.shape[0]/(X_train2.shape[0]+X_test2.shape[0])),
             X_test2.shape[0], (100*X_test2.shape[0]/(X_train2.shape[0]+X_test2.shape[0]))))

print('3. Regression imputation: \n\tThe sample size of training set : test set is {} ({:.2f}%) : {} ({:.2f}%).' 
      .format(X_train3.shape[0], (100*X_train3.shape[0]/(X_train3.shape[0]+X_test3.shape[0])),
             X_test3.shape[0], (100*X_test3.shape[0]/(X_train3.shape[0]+X_test3.shape[0]))))
```
```Markdown
1. Drop all missing: 
	The sample size of training set : test set is 206 (72.28%) : 79 (27.72%).
2. Mean imputation: 
	The sample size of training set : test set is 278 (71.47%) : 111 (28.53%).
3. Regression imputation: 
	The sample size of training set : test set is 278 (71.47%) : 111 (28.53%).
```
```Markdown
# Print the number of each diagnosis within each dataset
print('1. Drop all missing: \n\tNumber of instances of CN(class1), AD(class2), LMCI(class3) is: {}, {}, {}'
     .format(X_train1[y_train1==1].shape[0], X_train1[y_train1==2].shape[0], X_train1[y_train1==3].shape[0]))

print('2. Mean imputation: \n\tNumber of instances of CN(class1), AD(class2), LMCI(class3) is: {}, {}, {}'
     .format(X_train2[y_train2==1].shape[0], X_train2[y_train2==2].shape[0], X_train2[y_train2==3].shape[0]))

print('3. Regression imputation: \n\tNumber of instances of CN(class1), AD(class2), LMCI(class3) is: {}, {}, {}'
     .format(X_train3[y_train3==1].shape[0], X_train3[y_train3==2].shape[0], X_train3[y_train3==3].shape[0]))
```
```Markdown
1. Drop all missing: 
	Number of instances of CN(class1), AD(class2), LMCI(class3) is: 69, 43, 94
2. Mean imputation: 
	Number of instances of CN(class1), AD(class2), LMCI(class3) is: 79, 69, 130
3. Regression imputation: 
	Number of instances of CN(class1), AD(class2), LMCI(class3) is: 79, 69, 130
```
```Markdown
# Prepare for modeling
X_trains = [X_train1, X_train2, X_train3]
y_trains = [y_train1, y_train2, y_train3]
X_tests = [X_test1, X_test2, X_test3]
y_tests = [y_test1, y_test2, y_test3]
labels = ['Drop Missing', 'Mean Imputation', 'Regression Imputation']
```

### 2.Classification
#### 0) Principle Component Analysis (PCA)

```Markdown
fig, ax_pca1 = plt.subplots(1,3, figsize=(18,5))
ax_pca1 = ax_pca1.ravel()

fig, ax_pca2 = plt.subplots(1,3, figsize=(18,5))
ax_pca2 = ax_pca2.ravel()

for i in range(3):
    X_train = X_trains[i]
    y_train = y_trains[i]
    
    pca = PCA()
    x_train_pca = pca.fit_transform(X_train)

    # Create a dataset for the top two principle components
    train_pca = pd.DataFrame(y_train)
    train_pca['pc1'] = x_train_pca[:,0]
    train_pca['pc2'] = x_train_pca[:,1]

    # Generate a scatter plot for pc1 and pc2
    ax_pca1[i].scatter(train_pca[train_pca['y']==1]['pc1'], train_pca[train_pca['y']==1]['pc2'], label='CN')
    ax_pca1[i].scatter(train_pca[train_pca['y']==2]['pc1'], train_pca[train_pca['y']==2]['pc2'], label='AD')
    ax_pca1[i].scatter(train_pca[train_pca['y']==3]['pc1'], train_pca[train_pca['y']==3]['pc2'], label='LMCI')
    ax_pca1[i].set_xlabel('PC1')
    ax_pca1[i].set_ylabel('PC2')
    ax_pca1[i].legend(loc='best',frameon=True)
    ax_pca1[i].set_title('PCA (' + labels[i] + ')');

    # cumulative variance plot
    ax_pca2[i].plot(np.cumsum(pca.explained_variance_ratio_))
    ax_pca2[i].set_xlabel('Number of components',fontsize=13)
    ax_pca2[i].set_ylabel('Cumulative explained variance',fontsize=13)
    ax_pca2[i].set_title("Cumulative explained variance by principle components",fontsize=13);

    # Find the number of components to account for 80% of the variability in the feature set.
    print('({}) The number of PCs that account for 80% of the variance is {}.'
          .format(labels[i], 40-sum((np.cumsum(pca.explained_variance_ratio_)>=0.8).astype(int))+1))

    # Find the number of components to account for 90% of the variability in the feature set.
    print('({}) The number of PCs that account for 90% of the variance is {}.'
          .format(labels[i],40-sum((np.cumsum(pca.explained_variance_ratio_)>=0.9).astype(int))+1))
```
```Markdown
(Drop Missing) The number of PCs that account for 80% of the variance is 29.
(Drop Missing) The number of PCs that account for 90% of the variance is 33.
(Mean Imputation) The number of PCs that account for 80% of the variance is 29.
(Mean Imputation) The number of PCs that account for 90% of the variance is 33.
(Regression Imputation) The number of PCs that account for 80% of the variance is 29.
(Regression Imputation) The number of PCs that account for 90% of the variance is 33.
```
(/.png)

#### 1) Multinomial Logistic Modeling

