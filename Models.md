# Models
## Contents
[1. Data Preparation](#-1.-data-preparation)<br>
    [1) Reading and Cleaning Data](.#-1\)-reading-and-cleaning-data)<br>
    2) Imputing and Scaling Data
2. Classification
    0) Principle Component Analysis (PCA)
    1) Multinomial Logistic Modeling
    2) Linear discriminant analysis (LDA)
    3) Quadratic Discriminant Analysis (QDA)
    4) k-NN
    5) Decision Tree
    6) Bagging
    7) Random Forest
    8) AdaBoost
    
    

### 1. Data Preparation
#### 1) Reading and Cleaning Data

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

#### 2) Imputing and Scaling Data
**We used three imputation methods to impute missing data:**
1. Deleting all missing in predictors
2. Mean imputation
3. Regression imputation

```py
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

```py
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

```py
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

```py
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
```py
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
```py
# Prepare for modeling
X_trains = [X_train1, X_train2, X_train3]
y_trains = [y_train1, y_train2, y_train3]
X_tests = [X_test1, X_test2, X_test3]
y_tests = [y_test1, y_test2, y_test3]
labels = ['Drop Missing', 'Mean Imputation', 'Regression Imputation']
```

### 2.Classification
#### 0) Principle Component Analysis (PCA)

```py
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
![PC1 vs. PC2](/images/pca1.png)
![Cumulative explained variance](/images/pca2.png)

#### 1) Multinomial Logistic Modeling

```py
# Multinomial logistic model
logi_accs_train = []
logi_accs_test = []
logi_cvscores = []
logi_models = []

for i in range(3):
    X_train = X_trains[i]
    y_train = y_trains[i]
    X_test = X_tests[i]
    y_test = y_tests[i]
    
    logi_model = LogisticRegressionCV\
        (random_state=0, cv=5, penalty='l2', multi_class="auto", solver='lbfgs', max_iter=100).fit(X_train,y_train)
    logi_models.append(logi_model)
    
    # cross validation scores
    logi_cvscore = cross_val_score(logi_model, X_train, y_train, cv=5)
    logi_cvscores.append(logi_cvscore)
    
    # train and test accuracy of logistic regression
    logi_acc_train = logi_model.score(X_train, y_train)
    logi_acc_test = logi_model.score(X_test, y_test)
    logi_accs_train.append(logi_acc_train)
    logi_accs_test.append(logi_acc_test)
```
```py
# cross validation scores and accuracy
for i in range(3):
    print('({})'.format(labels[i]))
    print('Cross Validation Score of Multinomial Logistic Model: ', logi_cvscores[i])
    print('Mean of CV Score of Multinomial Logistic Model: {:.4f}' .format(logi_cvscores[i].mean()))
    print('Training accuracy: %.4f' % logi_accs_train[i])
    print('Test accuracy: %.4f' % logi_accs_test[i], '\n')
```
```Markdown
(Drop Missing)
Cross Validation Score of Multinomial Logistic Model:  [0.88095238 0.83333333 0.78571429 0.92682927 0.8974359 ]
Mean of CV Score of Multinomial Logistic Model: 0.8649
Training accuracy: 0.9563
Test accuracy: 0.8861 

(Mean Imputation)
Cross Validation Score of Multinomial Logistic Model:  [0.82142857 0.89285714 0.85714286 0.89285714 1.        ]
Mean of CV Score of Multinomial Logistic Model: 0.8929
Training accuracy: 0.9532
Test accuracy: 0.8829 

(Regression Imputation)
Cross Validation Score of Multinomial Logistic Model:  [0.85714286 0.89285714 0.82142857 0.83928571 0.94444444]
Mean of CV Score of Multinomial Logistic Model: 0.8710
Training accuracy: 0.9496
Test accuracy: 0.9099 
```
**We looked at the prediction accuracy of the best model (using regression imputation) on each diagnosis label, and we found they still remained high**
```py
# prediction accuracy of each label of regression imputation model
print(classification_report(y_trains[2], logi_models[2].predict(X_trains[2])))
print(classification_report(y_tests[2], logi_models[2].predict(X_tests[2])))
```
```Markdown
         label    Training Accuracy    Test Accuracy 
           1           0.97            	   0.92
           2           0.93                0.88
           3           0.95                0.91
```


#### 2) Linear discriminant analysis (LDA)
```py
# LDA
lda_accs_train = []
lda_accs_test = []
lda_models = []

for i in range(3):
    X_train = X_trains[i]
    y_train = y_trains[i]
    X_test = X_tests[i]
    y_test = y_tests[i]
    
    lda = LinearDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)
    lda_models.append(lda)
    lda_accs_train.append(lda.score(X_train, y_train))
    lda_accs_test.append(lda.score(X_test, y_test))
```
```py
# train and test accuracy
for i in range(3):
    print('({})'.format(labels[i]))
    print('Training accuracy of LDA model: %.4f' % lda_accs_train[i])
    print('Test accuracy of LDA model: %.4f' % lda_accs_test[i])
```
```Markdown
(Drop Missing)
Training accuracy of LDA model: 0.9126
Test accuracy of LDA model: 0.8987

(Mean Imputation)
Training accuracy of LDA model: 0.9281
Test accuracy of LDA model: 0.8559

(Regression Imputation)
Training accuracy of LDA model: 0.9137
Test accuracy of LDA model: 0.8739
```

#### 3) Quadratic Discriminant Analysis (QDA)
```py
# QDA
qda_accs_train = []
qda_accs_test = []
qda_models = []

for i in range(3):
    X_train = X_trains[i]
    y_train = y_trains[i]
    X_test = X_tests[i]
    y_test = y_tests[i]
    
    qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)
    qda_models.append(qda)
    qda_accs_train.append(qda.score(X_train, y_train))
    qda_accs_test.append(qda.score(X_test, y_test))
```
```py
# train and test accuracy
for i in range(3):
    print('({})'.format(labels[i]))
    print('Training accuracy of QDA model: %.4f' % qda_accs_train[i])
    print('Test accuracy of QDA model: %.4f' % qda_accs_test[i], '\n')
```
```Markdown
(Drop Missing)
Training accuracy of QDA model: 0.9806
Test accuracy of QDA model: 0.8354 

(Mean Imputation)
Training accuracy of QDA model: 0.9568
Test accuracy of QDA model: 0.8559 

(Regression Imputation)
Training accuracy of QDA model: 0.9424
Test accuracy of QDA model: 0.8739 
```

**We looked at the prediction accuracy of the best model (using regression imputation) on each diagnosis label, and we found they still remained high**
```py
# prediction accuracy of each label of regression imputation model
print(classification_report(y_trains[2], lda_models[2].predict(X_trains[2])))
print(classification_report(y_tests[2], lda_models[2].predict(X_tests[2])))
print(classification_report(y_trains[2], qda_models[2].predict(X_trains[2])))
print(classification_report(y_tests[2], qda_models[2].predict(X_tests[2])))
```
```Markdown
LDA:
         label    Training Accuracy    Test Accuracy 
           1           0.94            	   0.84
           2           0.90                0.93
           3           0.91                0.87
QDA:
         label    Training Accuracy    Test Accuracy 
           1           0.98            	   0.91
           2           0.89                0.75
           3           0.95                0.90
```


#### 4) k-NN
**First, we fit multiple k-NN models on training set, and find the best number of k based on cross validation scores**
```py
#Fit knn models on training set
fig, ax_knn = plt.subplots(1,3, figsize=(18,5))
ax_knn = ax_knn.ravel()

for i in range(3):
    X_train = X_trains[i]
    y_train = y_trains[i]
    X_test = X_tests[i]
    y_test = y_tests[i]
    
    knn_cvscores_avg = []
    for j in range(1,30):
        knn = KNeighborsClassifier(weights='uniform',n_neighbors=j)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        knn_cvscores_avg.append(scores.mean())

    ax_knn[i].plot(range(1,30), knn_cvscores_avg)
    ax_knn[i].set_xlabel("Neighbours-k")
    ax_knn[i].set_ylabel("Prediction Accuracy")
    ax_knn[i].set_title(labels[i]);

    print('({}) Best number of Neighbours-k: {}' .format(labels[i], knn_cvscores_avg.index(max(knn_cvscores_avg))+1))
```
```Markdown
(Drop Missing) Best number of Neighbours-k: 5
(Mean Imputation) Best number of Neighbours-k: 14
(Regression Imputation) Best number of Neighbours-k: 10
```
![knn_cvscores](/images/knn.png)

**Then we fit the best k-NN model for different imputation dataset**
```py
# fit the best kNN model for each dataset
knn_accs_train = []
knn_accs_test = []
ks = [5,14,10]
knn_models = []

for i in range(3):
    knn_best = KNeighborsClassifier(weights='uniform',n_neighbors=ks[i]).fit(X_trains[i],y_trains[i])
    knn_accs_train.append(knn_best.score(X_trains[i],y_trains[i]))
    knn_accs_test.append(knn_best.score(X_tests[i],y_tests[i]))
    knn_models.append(knn_best)

    print('({})'.format(labels[i]))
    print('Training accuracy of k-NN model (k={}): {:.4f}' .format(ks[i], knn_accs_train[i]))
    print('Test accuracy of k-NN model (k={}): {:.4f}' .format(ks[i], knn_accs_test[i]), '\n')
```
```Markdown
(Drop Missing)
Training accuracy of k-NN model (k=5): 0.8495
Test accuracy of k-NN model (k=5): 0.7975 

(Mean Imputation)
Training accuracy of k-NN model (k=14): 0.8345
Test accuracy of k-NN model (k=14): 0.8108 

(Regression Imputation)
Training accuracy of k-NN model (k=10): 0.8237
Test accuracy of k-NN model (k=10): 0.7928 
```

**We looked at the prediction accuracy of the best model (using mean imputation) on each diagnosis label, and we found they still remained high**
```py
# prediction accuracy of each label of mean imputation model
print(classification_report(y_trains[1], knn_models[1].predict(X_trains[1])))
print(classification_report(y_tests[1], knn_models[1].predict(X_tests[1])))
```
```Markdown
         label    Training Accuracy    Test Accuracy 
           1           0.86            	   0.79
           2           0.82                0.86
           3           0.82                0.80
```

#### 5) Decision Tree
**First, we fit multiple decision trees with different max tree depth on training set, and find the best max tree depth based on cross validation scores**
```py
# Fit Decision trees with different max tree depth

fig, axd = plt.subplots(1,3, figsize=(20,5))
axd = axd.ravel()

for i in range(3):
    
    X_train = X_trains[i]
    y_train = y_trains[i]
    X_test = X_tests[i]
    y_test = y_tests[i]
    
    dt_accuracy_dict = {}
    dt_cvscore_dict = {}
    
    for j in range(2,11):
        dt = DecisionTreeClassifier(random_state=0, max_depth = j)
        dt.fit(X_train, y_train)
        dt_accuracy_dict[j] = dt.score(X_train, y_train)                           # accuracy on training set
        dt_cvscore_dict[j] = cross_val_score(dt, X_train, y_train, cv=5).mean()    # mean score on validation set
    
    axd[i].set_title(labels[i])
    axd[i].plot(dt_accuracy_dict.keys(),dt_accuracy_dict.values(),color='orange',marker='o',label='Training Accuracy')
    axd[i].plot(dt_cvscore_dict.keys(),dt_cvscore_dict.values(),color='g',marker='o',label='CV score')
    axd[i].set_xlabel('Maximum tree depth')
    axd[i].set_ylabel('Accuracy')
    axd[i].legend(loc='upper left');
    
    best_depth = 0
    for k,v in dt_cvscore_dict.items():
        if v == max(dt_cvscore_dict.values()):
            best_depth = k
    print('({}) Optimal max tree depth = {}'. format(labels[i], best_depth))
```
```Markdown
(Drop Missing) Optimal max tree depth = 2
(Mean Imputation) Optimal max tree depth = 3
(Regression Imputation) Optimal max tree depth = 4
```
![dt_cvscores](/images/dt1.png)

**Then we fit the best decision tree for different imputation dataset**
```py
# Decision tree with depth = 2, 3, 4 for three datasets
dt_models = []
dt_accs_train = []
dt_accs_test = []
depths = [2, 3, 4]

for i in range(3):
    dt = DecisionTreeClassifier(max_depth = depths[i]).fit(X_trains[i], y_trains[i])
    dt_models.append(dt)
    # train and test accuracy
    dt_accs_train.append(dt.score(X_trains[i], y_trains[i]))
    dt_accs_test.append(dt.score(X_tests[i], y_tests[i]))
    
    print('({})'.format(labels[i]))
    print('Training accuracy of decision tree (max depth={}): {:.4f}' .format(depths[i], dt_accs_train[i]))
    print('Test accuracy of decision tree (max depth={}): {:.4f}' .format(depths[i], dt_accs_test[i]), '\n')
```
```Markdown
(Drop Missing)
Training accuracy of decision tree (max depth=2): 0.9126
Test accuracy of decision tree (max depth=2): 0.9367 

(Mean Imputation)
Training accuracy of decision tree (max depth=3): 0.9353
Test accuracy of decision tree (max depth=3): 0.9369 

(Regression Imputation)
Training accuracy of decision tree (max depth=4): 0.9460
Test accuracy of decision tree (max depth=4): 0.9189 
```

**We can have a look at the structure of each decision tree**
```py
# This code is adapted from
# http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
def show_tree_structure(clf):
    tree = clf.tree_

    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print(f"The binary tree structure has {n_nodes} nodes:\n")
    
    for i in range(n_nodes):
        indent = node_depth[i] * "  "
        if is_leaves[i]:
            prediction = clf.classes_[np.argmax(tree.value[i])]
            print(f"{indent}node {i}: predict class {prediction}")
        else:
            print("{}node {}: if X[:, {}] <= {:.3f} then go to node {}, else go to node {}".format(
                indent, i, feature[i], threshold[i], children_left[i], children_right[i]))
```
```py
# tree structure of models dropping missing values
show_tree_structure(dt_models[0])
```
```Markdown
The binary tree structure has 5 nodes:

node 0: if X[:, 11] <= -0.731 then go to node 1, else go to node 2
  node 1: predict class 1
  node 2: if X[:, 11] <= 0.897 then go to node 3, else go to node 4
    node 3: predict class 3
    node 4: predict class 2
```
```py
# tree structure of models using mean imputation
show_tree_structure(dt_models[1])
```
```Markdown
The binary tree structure has 9 nodes:

node 0: if X[:, 11] <= -0.837 then go to node 1, else go to node 2
  node 1: predict class 1
  node 2: if X[:, 11] <= 0.561 then go to node 3, else go to node 6
    node 3: if X[:, 4] <= -1.196 then go to node 4, else go to node 5
      node 4: predict class 2
      node 5: predict class 3
    node 6: if X[:, 4] <= -0.069 then go to node 7, else go to node 8
      node 7: predict class 2
      node 8: predict class 3
```
```py
# tree structure of models using regression imputation
show_tree_structure(dt_models[2])
```
```Markdown
The binary tree structure has 15 nodes:

node 0: if X[:, 11] <= -0.837 then go to node 1, else go to node 2
  node 1: predict class 1
  node 2: if X[:, 11] <= 0.561 then go to node 3, else go to node 10
    node 3: if X[:, 4] <= -1.196 then go to node 4, else go to node 7
      node 4: if X[:, 10] <= -0.255 then go to node 5, else go to node 6
        node 5: predict class 3
        node 6: predict class 2
      node 7: if X[:, 17] <= -1.758 then go to node 8, else go to node 9
        node 8: predict class 1
        node 9: predict class 3
    node 10: if X[:, 4] <= -0.069 then go to node 11, else go to node 14
      node 11: if X[:, 9] <= -0.534 then go to node 12, else go to node 13
        node 12: predict class 3
        node 13: predict class 2
      node 14: predict class 3
```

#### 6) Bagging
**We used the best decision tree for each imputation method as the base model, and do Bagging with 50 bootstrapping samples.**

```python
#do 50 bootstrapping and store the results in bagging_train and bagging_test 
bag_accs_train = []
bag_accs_test = []
tops_bagging = []

# function to get the prediction accuracys
def bagging_pred(df, indexs):
    df['pred'] = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        count1, count2, count3 = 0, 0, 0
        for j in range(n_trees):
            if df.iloc[i,j] == 1:
                count1 += 1
            elif df.iloc[i,j] == 2:
                count2 += 1
            elif df.iloc[i,j] == 3:
                count3 += 1
        counts = [count1, count2, count3]
        df.at[indexs[i], 'pred'] = counts.index(np.max(counts)) + 1
    return df

for i in range(3):
    
    X_train = X_trains[i]
    y_train = y_trains[i]
    X_test = X_tests[i]
    y_test = y_tests[i]
    
    #Creating model
    np.random.seed(0)
    simpletree = DecisionTreeClassifier(max_depth=depths[i])    # best tree of each dataset
    
    #Initializing variables
    n_trees = 50
    predictions_train = np.zeros((X_train.shape[0], n_trees))
    predictions_test = np.zeros((X_test.shape[0], n_trees))
    top_bagging = []

    data_train_n = pd.concat([X_train, y_train],axis=1)
    #Conduct bootstraping iterations
    for j in range(n_trees):
        temp = data_train_n.sample(frac=1, replace=True)
        boot_y = temp['y']
        boot_xx = temp[predictors]
        simpletree.fit(boot_xx, boot_y)  
        predictions_train[:,j] = simpletree.predict(X_train)   
        predictions_test[:,j] = simpletree.predict(X_test)
        
        # count the times each feature used at the top node in bagging
        top_bagging.append(simpletree.tree_.feature[0])
    
    tops_bagging.append(top_bagging)
    
    #Make Predictions Dataframe
    columns = ['bootstrap model ' + str(k+1) + "'s prediction" for k in range(n_trees)]
    indexs_train = ['training row ' + str(k+1) for k in range(len(X_train))]
    indexs_test = ['test row ' + str(k+1) for k in range(len(X_test))]
    bagging_train = pd.DataFrame(predictions_train, columns=columns, index=indexs_train)
    bagging_test = pd.DataFrame(predictions_test, columns=columns, index=indexs_test)
    
    # get the prediction accuracy
    bagging_train = bagging_pred(bagging_train, indexs_train)
    bagging_test = bagging_pred(bagging_test, indexs_test)
    
    bag_accs_train.append(accuracy_score(y_train, bagging_train.pred))
    bag_accs_test.append(accuracy_score(y_test, bagging_test.pred))
```
```python
# print accuracy
for i in range(3):
    print('({})'.format(labels[i]))
    print('Training accuracy of bagging with 50 trees (max depth={}): {:.4f}' .format(depths[i], bag_accs_train[i]))
    print('Test accuracy of bagging with 50 trees (max depth={}): {:.4f}' .format(depths[i], bag_accs_test[i]))
    print('Feature used most at the top node in bagging: 11 \n')
```
```Markdown
(Drop Missing)
Training accuracy of bagging with 50 trees (max depth=2): 0.9126
Test accuracy of bagging with 50 trees (max depth=2): 0.9367
Feature used most at the top node in bagging: 11 

(Mean Imputation)
Training accuracy of bagging with 50 trees (max depth=3): 0.9353
Test accuracy of bagging with 50 trees (max depth=3): 0.9369
Feature used most at the top node in bagging: 11 

(Regression Imputation)
Training accuracy of bagging with 50 trees (max depth=4): 0.9460
Test accuracy of bagging with 50 trees (max depth=4): 0.9459
Feature used most at the top node in bagging: 11 
```
**We looked at the prediction accuracy of the best model (using regression imputation) on each diagnosis label, and we found they still remained high**
```py
# prediction accuracy of each label of regression imputation model
print(classification_report(y_trains[2], dt_models[2].predict(X_trains[2])))
print(classification_report(y_tests[2], dt_models[2].predict(X_tests[2])))
```
```Markdown
         label    Training Accuracy    Test Accuracy 
           1           0.98            	   0.92
           2           0.91                0.91
           3           0.94                0.92
```

#### 7) Random Forest
**We used the best decision tree as the base model, built random forest models with different number of trees, and chose the optimal number of trees for each imputation method based on training accuracy.**
```python
# build random forest with different number of trees

fig, ax_rf = plt.subplots(1,3, figsize=(20,5))
ax_rf = ax_rf.ravel()

for i in range(3):
    X_train = X_trains[i]
    y_train = y_trains[i]
    X_test = X_tests[i]
    y_test = y_tests[i]
    
    scores_RF = []
    for j in range(1,100):
        RF = RandomForestClassifier(max_depth=depths[i], n_estimators=j, max_features='sqrt', random_state=0).\
        fit(X_train,y_train)
        scores_RF.append(accuracy_score(y_train, RF.predict(X_train)))
    
    ax_rf[i].plot(range(1,100), scores_RF)
    ax_rf[i].set_xlabel("Number of estimator")
    ax_rf[i].set_ylabel("Prediction Accuracy")
    ax_rf[i].set_title("Prediction Accuracy vs Number of estimators");

    print('({}) Optimal number of trees (max depth={}): {}' 
          .format(labels[i], depths[i], scores_RF.index(max(scores_RF))+1))
```
```Markdown
(Drop Missing) Optimal number of trees (max depth=2): 10
(Mean Imputation) Optimal number of trees (max depth=3): 59
(Regression Imputation) Optimal number of trees (max depth=4): 35
```
![Random Forest1](/images/dt2.png)

**Now we fit the best random forest for each dataset with optimal number of trees.**
```python
# best random forest for each dataset
rfs = [10, 59, 35]
rf_models = []
rf_accs_train = []
rf_accs_test = []
feature_importances_all = []
fig, ax_rf2 = plt.subplots(3,1, figsize=(10,20))
ax_rf2 = ax_rf2.ravel()

for i in range(3):
    RF = RandomForestClassifier(max_depth=depths[i], n_estimators=rfs[i], 
                                max_features='sqrt', random_state=0).fit(X_trains[i],y_trains[i])
    rf_models.append(RF)

    rf_accs_train.append(RF.score(X_trains[i], y_trains[i]))
    rf_accs_test.append(RF.score(X_tests[i], y_tests[i]))

    feature_importances = pd.DataFrame(RF.feature_importances_,
                                    columns=['importance']).sort_values('importance',ascending=False)
    feature_importances_all.append(feature_importances)
    
    names = X_trains[i].columns.get_values()
    importances =feature_importances.importance
    indices = np.argsort(importances)
    ax_rf2[i].set_title('Feature Importances (' + labels[i] + ')')
    ax_rf2[i].barh(range(len(indices)), importances[indices], color='b', align='center')
    ax_rf2[i].set_yticks(range(len(indices)))
    ax_rf2[i].set_yticklabels(names[indices])
    ax_rf2[i].set_xlabel('Relative Importance');
```
![Random Forest2](/images/rf1.png)
```python
# accuracys
for i in range(3):
    print('({})'.format(labels[i]))
    print('Training accuracy of a random forest with {} trees and max tree depth={} is : {:.4f}.' 
          .format(rfs[i], depths[i], rf_accs_train[i]))
    print('Test accuracy of a random forest with {} trees and max tree depth={} is : {:.4f}. \n' 
          .format(rfs[i], depths[i], rf_accs_test[i]))
```
```Markdown
(Drop Missing)
Training accuracy of a random forest with 10 trees and max tree depth=2 is : 0.9175.
Test accuracy of a random forest with 10 trees and max tree depth=2 is : 0.9367. 

(Mean Imputation)
Training accuracy of a random forest with 59 trees and max tree depth=3 is : 0.9317.
Test accuracy of a random forest with 59 trees and max tree depth=3 is : 0.9369. 

(Regression Imputation)
Training accuracy of a random forest with 35 trees and max tree depth=4 is : 0.9532.
Test accuracy of a random forest with 35 trees and max tree depth=4 is : 0.9640. 
```
**We looked at the prediction accuracy of the best model (using regression imputation) on each diagnosis label, and we found they still remained high**
```py
# prediction accuracy of each label of regression imputation model
print(classification_report(y_trains[2], rf_models[2].predict(X_trains[2])))
print(classification_report(y_tests[2], rf_models[2].predict(X_tests[2])))
```
```Markdown
         label    Training Accuracy    Test Accuracy 
           1           0.97            	   0.97
           2           0.93                0.95
           3           0.95                0.97
```

#### 8) AdaBoost
**We used the best decision tree for each imputation method as the base model to build AdaBoost model.**
```py
# Adaboost model using decision tree as baseline model
ada_accs_train = []
ada_accs_test = []
ada_models = []
fig, ax_ada = plt.subplots(1,3, figsize=(20,5))
ax_ada = ax_ada.ravel()

for i in range(3):
    X_train = X_trains[i]
    y_train = y_trains[i]
    X_test = X_tests[i]
    y_test = y_tests[i]
    
    #Training
    ada_dt = AdaBoostClassifier(base_estimator=DecisionTreeClassifier
                                (max_depth=depths[i],random_state=0),learning_rate=0.05)
    ada_dt.fit(X_train, y_train)
    ada_models.append(ada_dt)
    
    ada_accs_train.append(ada_dt.score(X_train, y_train))
    ada_accs_test.append(ada_dt.score(X_test, y_test))

    #Plot Iteration based score
    train_scores_ada_dt = list(ada_dt.staged_score(X_train,y_train))
    test_scores_ada_dt = list(ada_dt.staged_score(X_test, y_test))

    ax_ada[i].plot(train_scores_ada_dt, label='training set')
    ax_ada[i].plot(test_scores_ada_dt, label='test test')
    ax_ada[i].set_xlabel('Number of Iteration')
    ax_ada[i].set_ylabel('Prediction Accuracy')
    ax_ada[i].set_title(labels[i])
    ax_ada[i].legend(frameon=True);
```
![AdaBoost](/images/rf2.png)
```python
# accuracys
for i in range(3):
    print('({})'.format(labels[i]))
    print('Training accuracy of Ada boosting model with max tree depth={} is : {:.4f}.' 
          .format(depths[i], ada_accs_train[i]))
    print('Test accuracy of Ada boosting model with max tree depth={} is : {:.4f}. \n' 
          .format(depths[i], ada_accs_test[i]))
```
```Markdown
(Drop Missing)
Training accuracy of Ada boosting model with max tree depth=2 is : 0.9126.
Test accuracy of Ada boosting model with max tree depth=2 is : 0.8987. 

(Mean Imputation)
Training accuracy of Ada boosting model with max tree depth=3 is : 0.9209.
Test accuracy of Ada boosting model with max tree depth=3 is : 0.9009. 

(Regression Imputation)
Training accuracy of Ada boosting model with max tree depth=4 is : 1.0000.
Test accuracy of Ada boosting model with max tree depth=4 is : 0.8919. 
```
**We looked at the prediction accuracy of the best model (using regression imputation) on each diagnosis label, and we found they still remained high, expect for the accuracy of the second label (AD).**
```py
# prediction accuracy of each label of regression imputation model
print(classification_report(y_trains[1], ada_models[1].predict(X_trains[1])))
print(classification_report(y_tests[1], ada_models[1].predict(X_tests[1])))
```
```Markdown
         label    Training Accuracy    Test Accuracy 
           1           1.00            	   0.97
           2           0.81                0.78
           3           0.92                0.91
```
