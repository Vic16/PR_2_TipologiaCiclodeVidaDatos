import pandas as pd
import numpy as np
from scipy.stats import levene
from scipy.stats import fligner
from scipy.stats import chi2_contingency
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from statsmodels.stats.proportion import test_proportions_2indep
from joblib import dump, load

# Añadimos al dataset el nombre correspondiente de cada columna
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang','oldpeak', 'slope', 'ca', 'thal', 'num']
df = pd.read_csv("data/processed.cleveland.csv", header=None, names=columns)
df.head(2)

df = df.apply(pd.to_numeric, errors='coerce')
df.info()


df = df.replace({'cp': {4: 0},
                 'thal':{3 : "normal",6.0 : "fixed defect",7.0 : "reversable defect"},
                 'num':{2 : 1,3 : 1,4 : 1}})
df.sample(2)
df.info()

df = df.replace({'cp': {0: 'typical angina',1: 'atypical angina',2: 'non-anginal pain',3: 'asymptomatic'},
                 'sex':{1:'Male',0:'Female'},
                 'restecg':{0:'Normal',1:'ST-T wave abnormality',2:'Left ventricular hypertrophy'},
                 'exang':{0:'Yes Angina',1:'No Angina'},
                 'fbs':{0:'fastingBloodSugar_false',1:'fastingBloodSugar_true'},
                 'slope':{1:'upsloping',2:'flat',3:'downsloping'},})
df.to_csv("data/dataset_proc.csv", index=False, sep="|")
df.sample(2)

df.dropna(inplace=True)

fig, axes = plt.subplots(2, 3, figsize=(15,7))
sns.boxplot(ax=axes[0,0], y="age", palette="viridis", data=df).set(title='Distribución de la edad')
sns.boxplot(ax=axes[0,1], y="trestbps", palette="viridis", data=df).set(title='Distribución de la presión sanguínea')
sns.boxplot(ax=axes[0,2], y="chol" ,palette="viridis", data=df).set(title='Distribución del colesterol')

sns.boxplot(ax=axes[1,0], y="thalach" ,palette="viridis", data=df).set(title='Distribución del ratio máximo de pulsasiones')
sns.boxplot(ax=axes[1,1], y="oldpeak" ,palette="viridis", data=df).set(title='Distribución de la frecuencia cardíaca máxima')
sns.boxplot(ax=axes[1,2], y="ca" ,palette="viridis", data=df).set(title='Distribución de los vasos')

plt.show()

fig, axes = plt.subplots(2, 3, figsize=(20,9))

sns.histplot(ax=axes[0,0], x="age", palette="viridis", data=df).set(title='Distribución de la edad')
sns.histplot(ax=axes[0,1], x="trestbps", palette="viridis", data=df).set(title='Distribución de la presión sanguínea')
sns.histplot(ax=axes[0,2], x="chol" ,palette="viridis", data=df).set(title='Distribución del colesterol')

sns.histplot(ax=axes[1,0], x="thalach" ,palette="viridis", data=df).set(title='Distribución del ratio máximo de pulsasiones')
sns.histplot(ax=axes[1,1], x="oldpeak" ,palette="viridis", data=df).set(title='Distribución de la frecuencia cardíaca máxima')
sns.histplot(ax=axes[1,2], x="ca" ,palette="viridis", data=df, bins=4).set(title='Distribución de los vasos')
plt.show()

df.describe()

def replaceOuliers(column):
    """
    """
    colReplace = np.array(column)
    median = np.median(column)

    upper =  np.percentile(np.array(column),95)
    lower =  np.percentile(np.array(column),5)
    colReplace[colReplace[:] > upper] = median
    colReplace[colReplace[:] < lower] = median

    return list(colReplace)

df["trestbps"] = replaceOuliers(list(df["trestbps"]))
df["chol"] = replaceOuliers(list(df["chol"]))
df["thalach"] = replaceOuliers(list(df["thalach"]))
df["oldpeak"] = replaceOuliers(list(df["oldpeak"]))

df[["age","trestbps","chol","thalach","oldpeak","ca"]].describe()

fig, axes = plt.subplots(2, 2, figsize=(20,9))
sns.histplot(x=df["age"], ax=axes[0,0]).set(title='Distribución de la edad de los pacientes')
sns.histplot(x=df["age"], hue=df["num"], ax=axes[0,1]).set(title='Distribución de la edad de los pacientes en función de la presencia de enfermedad')
sns.boxplot(y=df["age"], ax=axes[1,0]).set(title='Distribución de la edad de los pacientes')
sns.boxplot(y=df["age"], x=df["num"], ax=axes[1,1]).set(title='Distribución de la edad de los pacientes en función de la presencia de enfermedad')
plt.show()

df[["age","num"]].groupby("num").describe().T

fig, axes = plt.subplots(1, 2, figsize=(20,4))
sns.countplot(x=df["sex"], ax=axes[0]).set(title='Frecuencias del género')
sns.countplot(x=df["sex"], hue=df["num"], ax=axes[1]).set(title='Frecuencias del género en función de la presencia de enfermedad')
plt.show()

pd.crosstab(index= df["sex"], columns=df["num"], normalize=True)

fig, axes = plt.subplots(1, 2, figsize=(20,4))
sns.countplot(x=df["cp"], ax=axes[0]).set(title='Frecuencias de los tipos de dolor en el pecho')
sns.countplot(x=df["cp"], hue=df["num"], ax=axes[1]).set(title='Frecuencias de los tipos de dolor en el pecho en función de la presencia de enfermedad')
plt.show()

pd.crosstab(index= df["cp"], columns=df["num"], normalize=True)

fig, axes = plt.subplots(2, 2, figsize=(20,9))

sns.histplot(x=df["trestbps"], ax=axes[0,0]).set(title='Distribución de la presión sanguínea')
sns.histplot(x=df["trestbps"], hue=df["num"], ax=axes[0,1]).set(title='Distribución de la la presión sanguínea en función de la presencia de enfermedad')

sns.boxplot(y=df["trestbps"], ax=axes[1,0]).set(title='Distribución de la presión sanguínea')
sns.boxplot(y=df["trestbps"], x=df["num"], ax=axes[1,1]).set(title='Distribución de la presión sanguínea en función de la presencia de enfermedad')

plt.show()

df[["trestbps","num"]].groupby("num").describe().T

fig, axes = plt.subplots(2, 2, figsize=(20,9))

sns.histplot(x=df["chol"], ax=axes[0,0]).set(title='Distribución del colesterol')
sns.histplot(x=df["chol"], hue=df["num"], ax=axes[0,1]).set(title='Distribución del colesterol en función de la presencia de enfermedad')

sns.boxplot(y=df["chol"], ax=axes[1,0]).set(title='Distribución del colesterol')
sns.boxplot(y=df["chol"], x=df["num"], ax=axes[1,1]).set(title='Distribución del colesterol en función de la presencia de enfermedad')

plt.show()

df[["chol","num"]].groupby("num").describe().T

fig, axes = plt.subplots(1, 2, figsize=(20,4))

sns.countplot(x=df["fbs"], ax=axes[0]).set(title='Frecuencias de Fbs')

sns.countplot(x=df["fbs"], hue=df["num"], ax=axes[1]).set(title='Frecuencias de Fbs en función de la presencia de enfermedad')

plt.show()

pd.crosstab(index= df["fbs"], columns=df["num"], normalize=True)
fig, axes = plt.subplots(1, 2, figsize=(20,4))

sns.countplot(x=df["restecg"], ax=axes[0]).set(title='Frecuencias de restecg')

sns.countplot(x=df["restecg"], hue=df["num"], ax=axes[1]).set(title='Frecuencias de restecg en función de la presencia de enfermedad')

plt.show()
pd.crosstab(index= df["restecg"], columns=df["num"], normalize=True)

fig, axes = plt.subplots(2, 2, figsize=(20,9))

sns.histplot(x=df["thalach"], ax=axes[0,0]).set(title='Distribución de la presión sanguínea')
sns.histplot(x=df["thalach"], hue=df["num"], ax=axes[0,1]).set(title='Distribución de la la presión sanguínea en función de la presencia de enfermedad')

sns.boxplot(y=df["thalach"], ax=axes[1,0]).set(title='Distribución de la presión sanguínea')
sns.boxplot(y=df["thalach"], x=df["num"], ax=axes[1,1]).set(title='Distribución de la presión sanguínea en función de la presencia de enfermedad')

plt.show()

fig, axes = plt.subplots(1, 2, figsize=(20,4))
sns.countplot(x=df["exang"], ax=axes[0]).set(title='Frecuencias de exang')
sns.countplot(x=df["exang"], hue=df["num"], ax=axes[1]).set(title='Frecuencias de exang en función de la presencia de enfermedad')
plt.show()

pd.crosstab(index= df["exang"], columns=df["num"], normalize=True)


fig, axes = plt.subplots(2, 2, figsize=(20,9))

sns.histplot(x=df["oldpeak"], ax=axes[0,0]).set(title='Distribución de oldpeak')
sns.histplot(x=df["oldpeak"], hue=df["num"], ax=axes[0,1]).set(title='Distribución de oldpeak en función de la presencia de enfermedad')

sns.boxplot(y=df["oldpeak"], ax=axes[1,0]).set(title='Distribución del oldpeak')
sns.boxplot(y=df["oldpeak"], x=df["num"], ax=axes[1,1]).set(title='Distribución de oldpeak en función de la presencia de enfermedad')

plt.show()

df[["oldpeak","num"]].groupby("num").describe().T

fig, axes = plt.subplots(1, 2, figsize=(20,3))

sns.countplot(x=df["slope"], ax=axes[0]).set(title='Frecuencias de slope')

sns.countplot(x=df["slope"], hue=df["num"], ax=axes[1]).set(title='Frecuencias de slope en función de la presencia de enfermedad')

plt.show()

pd.crosstab(index= df["slope"], columns=df["num"], normalize=True)

fig, axes = plt.subplots(2, 2, figsize=(20,9))

sns.histplot(x=df["ca"], ax=axes[0,0], bins=4).set(title='Distribución de ca')
sns.histplot(x=df["ca"], hue=df["num"], ax=axes[0,1], bins=4).set(title='Distribución de ca en función de la presencia de enfermedad')

sns.boxplot(y=df["ca"], ax=axes[1,0]).set(title='Distribución del oldpeak')
sns.boxplot(y=df["ca"], x=df["num"], ax=axes[1,1]).set(title='Distribución de ca en función de la presencia de enfermedad')

plt.show()

df[["ca","num"]].groupby("num").describe().T

fig, axes = plt.subplots(1, 2, figsize=(10,5))
sns.countplot(x=df["thal"], ax=axes[0]).set(title='Frecuencias de thal')
sns.countplot(x=df["thal"], hue=df["num"], ax=axes[1]).set(title='Frecuencias de thal en función de la presencia de enfermedad')
plt.show()

pd.crosstab(index= df["thal"], columns=df["num"], normalize=True)


# Test de normalidad Shapiro-Wilk
def testNormality(x):
    """
    x: list type to check normality

    """
    a = np.array(x)
    stat, p = stats.shapiro(x)
    if p < 0.05:
        print("Distribution is not normal, p value: {}".format(p))
    else:
        print("Distribution is normal, p value: {}".format(p))


# Test de varianzas Levene en variables con distribución normal
def testLevene(listA, listB):
    """
    listA: list type to compare variance
    listB: list type to compare variance

    """
    a = np.array(listA)
    b = np.array(listB)
    stat, p = levene(a, b)
    if p < 0.05:
        print("Variances not equal, p value: {}".format(p))
    else:
        print("Variances equal, p value: {}".format(p))

# Test de varianzas Fligner-Killeen en variables con distribución no normal
def testFligner(listA, listB):
    """
    listA: list type to compare variance
    listB: list type to compare variance

    """
    a = np.array(listA)
    b = np.array(listB)
    stat, p = fligner(a, b)
    if p < 0.05:
        print("Variances not equal, p value: {}".format(p))
    else:
        print("Variances equal, p value: {}".format(p))

df_sick = df[df["num"] == 1]
df_healthy = df[df["num"] == 0]


print("\nTest de normalidad para la edad en población con patologías cardiacas:")
testNormality(df_sick["age"])
print("\nTest de normalidad para la edad en población sin patologías cardiacas:")
testNormality(df_healthy["age"])

print(len(df_sick))
print(len(df_healthy))

print("Test de varianza entre la edad en pacientes con patologías caridacas y sin ellas:")
testFligner(df_sick["age"], df_healthy["age"])

testNormality(df_healthy["age"])

print("\nTest de normalidad para la presión sanguínea en reposo en población con patologías cardiacas:")
testNormality(df_sick["trestbps"])
print("\nTest de normalidad para la presión sanguínea en reposo en población sin patologías cardiacas:")
testNormality(df_healthy["trestbps"])

print("Test de varianza entre la presión sanguínea en reposo en pacientes con patologías caridacas y sin ellas:")
testFligner(df_sick["trestbps"], df_healthy["trestbps"])

print("\nTest de normalidad para el colesterol en población con patologías cardiacas:")
testNormality(df_sick["chol"])
print("\nTest de normalidad para el colesterol en población sin patologías cardiacas:")
testNormality(df_healthy["chol"])


print("Test de varianza entre el colesterol en pacientes con patologías caridacas y sin ellas:")
testFligner(df_sick["chol"], df_healthy["chol"])

print("\nTest de normalidad para la frecuencia cardiaca máxima en población con patologías cardiacas:")
testNormality(df_sick["thalach"])
print("\nTest de normalidad para la frecuencia cardiaca máxima en población sin patologías cardiacas:")
testNormality(df_healthy["thalach"])


print("Test de varianza entre la frecuencia cardiaca máxima en pacientes con patologías caridacas y sin ellas:")
testFligner(df_sick["thalach"], df_healthy["thalach"])


print("\nTest de normalidad para la depresión del ST inducida por el ejercicio en población con patologías cardiacas:")
testNormality(df_sick["oldpeak"])
print("\nTest de normalidad para la depresión del ST inducida por el ejercicio en población sin patologías cardiacas:")
testNormality(df_healthy["oldpeak"])

print("Test de varianza entre la frecuencia cardiaca máxima en pacientes con patologías caridacas y sin ellas:")
testFligner(df_sick["oldpeak"], df_healthy["oldpeak"])

print("\nTest de normalidad para el número de vasos principales en población con patologías cardiacas:")
testNormality(df_sick["ca"])
print("\nTest de normalidad para el número de vasos principales en población sin patologías cardiacas:")
testNormality(df_healthy["ca"])

print("Test de varianza entre el número de vasos principales en pacientes con patologías caridacas y sin ellas:")
testFligner(df_sick["ca"], df_healthy["ca"])


chiTest = chi2_contingency(np.array(pd.crosstab(index= df["cp"], columns=df["num"])))

print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s'%chiTest)
# No hace falta aclarar 'equal_var', ya que 'True' es la opción por
stats.ttest_ind(df_sick["chol"], df_healthy["chol"],
                alternative = 'greater')

pd.crosstab(index= df["sex"], columns=df["num"],margins=True)
testProp = test_proportions_2indep(count1=25, nobs1=96, count2=112, nobs2=201, alternative="two-sided")
testProp.pvalue

df[["age","trestbps","chol","thalach","ca"]].corr()

sns.heatmap(df[["age","trestbps","chol","thalach","ca"]].corr(),
            vmin=-1, vmax=1, center=0,
            square=True,
            annot=True,
            fmt=".1f"
           )
plt.show()

sns.pairplot(df[["age","trestbps","chol","thalach","ca"]],height=1.5,aspect=1.5)

df = pd.get_dummies(df, prefix='sex', columns=["sex"], drop_first=True)
df = pd.get_dummies(df, prefix='cp', columns=["cp"], drop_first=True)
df = pd.get_dummies(df, prefix='fbs', columns=["fbs"], drop_first=True)
df = pd.get_dummies(df, prefix='restecg', columns=["restecg"], drop_first=True)
df = pd.get_dummies(df, prefix='exang', columns=["exang"], drop_first=True)
df = pd.get_dummies(df, prefix='slope', columns=["slope"], drop_first=True)
df = pd.get_dummies(df, prefix='thal', columns=["thal"], drop_first=True)

df.shape
logistcRegresion = LogisticRegression()
decisionTree = DecisionTreeClassifier()



############### LR
logistcRegresion_params = {'C': [0.01,0.1,1,10,100,1000]}
############### DT
decisionTree_params = {"criterion":["gini","entropy"],"max_depth":[5,10,20]}



X = df.loc[:, df.columns != 'num']
y = list(df["num"])

X.head(2)
y[0:3]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model_GS_logisticRegression = GridSearchCV(estimator=logistcRegresion,
                                           param_grid=logistcRegresion_params,
                                           scoring = ['accuracy', 'f1','roc_auc'],
                                           n_jobs=-1,cv=5,verbose=False,refit='f1')
model_GS_logisticRegression.fit(X_scaled,y)
estimator = model_GS_logisticRegression.best_estimator_

dump(estimator, str("models/" + "bestModeloRegressionLogistc" + ".joblib"))

resultsRegressionLogistic = pd.DataFrame(model_GS_logisticRegression.cv_results_)
resultsRegressionLogistic = resultsRegressionLogistic[["mean_fit_time",
                                                       "params","mean_test_accuracy","std_test_accuracy",
                                                       "mean_test_f1","std_test_f1","mean_test_roc_auc","std_test_roc_auc"]]

resultsRegressionLogistic.head(2)
model_GS_decisionTree = GridSearchCV(estimator=decisionTree,
                                           param_grid=decisionTree_params,
                                           scoring = ['accuracy', 'f1','roc_auc'],
                                           n_jobs=-1,cv=5,verbose=False,refit='f1')

model_GS_decisionTree.fit(X_scaled,y)
estimator_dt = model_GS_decisionTree.best_estimator_

dump(estimator_dt, str("models/" + "bestModeloDecisionTree" + ".joblib"))

results_dt = pd.DataFrame(model_GS_decisionTree.cv_results_)
results_dt = results_dt[["mean_fit_time","params","mean_test_accuracy",
                         "std_test_accuracy","mean_test_f1","std_test_f1","mean_test_roc_auc","std_test_roc_auc"]]
results_dt.head(2)

resultsRegressionLogistic["Model"] = "Regresion Logistica"
results_dt["Model"] = "Decision Tree"
results = pd.concat([resultsRegressionLogistic, results_dt])
results.to_csv("resultsModels.csv", index=False, sep="|")

results[["Model","mean_fit_time"]].groupby("Model").mean()
results[["Model","mean_test_accuracy","mean_test_f1","mean_test_roc_auc"]].groupby("Model").mean()

model_RL = load("models/bestModeloRegressionLogistc.joblib")
model_DT = load("models/bestModeloDecisionTree.joblib")


y_pred_RL = model_RL.predict(X_scaled)
print(classification_report(y, y_pred_RL))

y_pred_DT = model_DT.predict(X_scaled)
print(classification_report(y, y_pred_DT))


ns_probs = [0 for _ in range(len(y))]
RL_props = model_RL.predict_proba(X_scaled)
RL_props = RL_props[:, 1]
DT_props = model_DT.predict_proba(X_scaled)
DT_props = DT_props[:, 1]
ns_auc = roc_auc_score(y, ns_probs)
RL_auc = roc_auc_score(y, RL_props)
DT_auc = roc_auc_score(y, DT_props)

print('Regresion Logistica: ROC AUC = %.3f' % (RL_auc))
print('Decision Tree: ROC AUC = %.3f' % (DT_auc))

ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
rL_fpr, rL_tpr, _ = roc_curve(y, RL_props)
dt_fpr, dt_tpr, _ = roc_curve(y, DT_props)

plt.plot(ns_fpr, ns_tpr, linestyle='--')
plt.plot(rL_fpr, rL_tpr, marker='.', label='Regresion Logistica')
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend()

for i,v,o in zip(importance, list(range(0,len(X.columns))) ,list(X.columns)):
    print('Feature {}: {}, Score: %.5f'.format(v,o) % (i))
plt.bar([x for x in range(len(importance))], importance,)
plt.xlim([0, 18])
plt.show()
