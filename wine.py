#Tratamiento de datos
import pandas as pd
import numpy as np

#Graficas
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#Modelado
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import sklearn.linear_model as linear_model

nombres = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide',
           'density','pH','sulphates','alcohol','quality']
ruta = 'C:/Users/hlope/Downloads/winequality-red.csv'
df = pd.read_csv(ruta, sep=';',names=nombres)


#for para describir algunos estadisticos manualmente
for column in df:
    #print(df[column].shape(), "<-- shape de ", column)
    print("<-------------------",column,"------------------->")
    print(df[column].dtype, "<-- type de ", column)
    print(df[column].std(),"<-- std")
    print(df[column].mean(), "<-- mean")
    print(df[column].median(), "<-- median")
    print(df[column].max(), "<-- max")


pd.options.display.max_columns = None

descripcion = df.describe().transpose()

#print(descripcion)

#df.plot(x="fixed acidity", y="pH")

"""#histogramas, 4 x ventana
fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
ax0.hist(x=df['fixed acidity'])
ax0.set_title('Fixed acidity')

ax1.hist(x=df['volatile acidity'])
ax1.set_title('Volatile acidity')

ax2.hist(x=df['citric acid'])
ax2.set_title('Citric acid')

ax3.hist(x=df['residual sugar'])
ax3.set_title('Residual sugar')
plt.show()

fig, ((ax4,ax5),(ax6,ax7)) = plt.subplots(2,2)
ax4.hist(x=df['chlorides'])
ax4.set_title('Chlorides')

ax5.hist(x=df['free sulfur dioxide'])
ax5.set_title('Free sulfur dioxide')

ax6.hist(x=df['total sulfur dioxide'])
ax6.set_title('Total sulfur dioxide')

ax7.hist(x=df['density'])
ax0.set_title('Density')

plt.show()

fig, ((ax8,ax9),(ax10,ax11)) = plt.subplots(2,2)
ax8.hist(x=df['pH'])
ax8.set_title('pH')

ax9.hist(x=df['sulphates'])
ax9.set_title('Sulphates')

ax10.hist(x=df['alcohol'])
ax10.set_title('Alcohol')

ax11.hist(x=df['quality'])
ax11.set_title('Quality')
plt.show()"""

#Heatmap de la correlación, mio
"""sns.heatmap(df, annot=True, annot_kws={"size": 7})
sns.heatmap(df.corr(), annot=True)
plt.show()"""



#heatmap. profesor
plt.figure(2)
mask = np.triu(np.ones_like(df.corr()))
heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Wine Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);
plt.show()

#matriz de correlaciones
"""correlaciones = df.corr()
print(correlaciones)"""

#pairplot de todas las variables, profesor
"""sns.pairplot(df)
plt.show(block=False)
input("Waiting")"""

#scatter
"""plt.scatter(df['volatile acidity'],df['quality'])
plt.show()
plt.plot(df['volatile acidity'],df['quality'], 'o')
plt.plot(df['quality'],df['volatile acidity'], 'o')
plt.show()"""

#plt.plot('fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol')


#estandarizar

#DataFrame con las 2 variables con más correlación respecto a calidad
dfC = pd.DataFrame(df['quality'])
dfC['alcohol'] = df['alcohol']
dfC['volatile acidity'] = df['volatile acidity']
dfC['total sulfur dioxide'] = df['total sulfur dioxide']
dfC['pH'] = df['pH']

"""#estandarización con min,max al df2
min_max_scaler = preprocessing.MinMaxScaler()
df2_minmax = min_max_scaler.fit_transform(df2)
df2_minmax = pd.DataFrame(df2_minmax)

#estandarizacion con z al df4
z_scaler = preprocessing.StandardScaler()
df3_z = z_scaler.fit_transform(df3)
df3_z = pd.DataFrame(df3_z)"""


#SIN MODIFICACIONES------------------
print("-----------------------------------------")
X = df.drop(columns='quality')
y = df['quality']


X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1), test_size=0.2, random_state=3)

lr_multiple = linear_model.LinearRegression()

lr_multiple.fit(X_train, y_train)

Y_pred_multiple = lr_multiple.predict(X_test)

print('Coeficients: ', lr_multiple.coef_)

print('\nIntercepts: ',lr_multiple.intercept_)

print('\nScore: ', lr_multiple.score(X_train, y_train))


#tirar calidad
X = dfC.drop(columns='quality')
y = dfC['quality']


X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1), test_size=0.2, random_state=3)

lr_multiple = linear_model.LinearRegression()

lr_multiple.fit(X_train, y_train)

Y_pred_multiple = lr_multiple.predict(X_test)

print('Coeficients: ', lr_multiple.coef_)

print('\nIntercepts: ',lr_multiple.intercept_)

print('\nPrecision: ', lr_multiple.score(X_train, y_train))



#define response variable
y = y_train

#define predictor variables
x = X_train

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())



#evaluar la suma de los errores
#por una prediccion, hay un epsilon
#entre más predicciones hagamos, se van sumando nuestros errores epsilons

#por cada 100 muestreos, el p-value cae dentro del rango establecido

#estandarizar/normalizar

#min max, clase pasada
"""min_max_scaler = preprocessing.MinMaxScaler()
dfC_minmax = min_max_scaler.fit_transform(dfC)
dfC_minmax = pd.DataFrame(dfC_minmax)

h = dfC_minmax.plot.hist()
print(h)
plt.show()"""


#zscore, da precision 1 ¿?
zscore_df = stats.zscore(dfC, axis=1)

zscore_df.columns = ['quality', 'alcohol','volatile acidity']

"""dfC_minmax.columns = ['quality', 'alcohol','volatile acidity']

X = dfC_minmax.drop(columns='quality')
y = dfC_minmax['quality']"""

X = zscore_df[['alcohol', 'volatile acidity']]
y = zscore_df['quality']




#graficas
"""print(zscore_df)
h = zscore_df.plot.hist()
print(h)
plt.show()"""

X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1), test_size=0.2, random_state=3)

lr_multiple = linear_model.LinearRegression()

lr_multiple.fit(X_train, y_train)

Y_pred_multiple = lr_multiple.predict(X_test)

print('Coeficients: ', lr_multiple.coef_)

print('\nIntercepts: ',lr_multiple.intercept_)

print("Z")
print('\nPrecision: ', lr_multiple.score(X_train, y_train))


#z scale, clase pasada
"""z_scaler = preprocessing.StandardScaler()
dfC_z = z_scaler.fit_transform(dfC_minmax)
dfC_z = pd.DataFrame(dfC_z)

dfC_z.columns = ['quality', 'alcohol','volatile acidity']

X = dfC_z.drop(columns='quality')
y = dfC_z['quality']


X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1), test_size=0.2, random_state=3)

lr_multiple = linear_model.LinearRegression()

lr_multiple.fit(X_train, y_train)

Y_pred_multiple = lr_multiple.predict(X_test)

print('Coeficients: ', lr_multiple.coef_)

print('\nIntercepts: ',lr_multiple.intercept_)

print('\nPrecision: ', lr_multiple.score(X_train, y_train))"""



#definir funcion para hacer min max
def minmax_rnorm(df):
    return (df - df.min()) / (df.max()-df.min())

df_min_max_norm = minmax_rnorm(dfC)




#graficas
"""print(df_min_max_norm)
h2 = df_min_max_norm.plot.hist()
print(h2)
plt.show()"""

X = df_min_max_norm[['alcohol', 'volatile acidity']]
y = df_min_max_norm['quality']


X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1), test_size=0.2, random_state=3)

lr_multiple = linear_model.LinearRegression()

lr_multiple.fit(X_train, y_train)

Y_pred_multiple = lr_multiple.predict(X_test)

print('Coeficients: ', lr_multiple.coef_)

print('\nIntercepts: ',lr_multiple.intercept_)

print("MinMax")
print('\nPrecision: ', lr_multiple.score(X_train, y_train))
#Da precision de 30, casi igual


#definir funcion para normalizar
def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

df_mean_norm = mean_norm(df_min_max_norm)

X = df_mean_norm[['alcohol', 'volatile acidity']]
y = df_mean_norm['quality']

X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1), test_size=0.2, random_state=3)

lr_multiple = linear_model.LinearRegression()

lr_multiple.fit(X_train, y_train)

Y_pred_multiple = lr_multiple.predict(X_test)

print('Coeficients: ', lr_multiple.coef_)

print('\nIntercepts: ',lr_multiple.intercept_)

print("ZZZZ")
print('\nPrecision: ', lr_multiple.score(X_train, y_train))
#igual da precision 0.3







df_nvo = pd.DataFrame(data= df['quality'])
df_nvo['alcohol'] = df['alcohol']
df_nvo['sulphates'] = df['sulphates']
df_nvo['volatile acidity']=df['volatile acidity']
df_nvo['chlorides']=df['chlorides']

df_nvo_mm = minmax_rnorm(df_nvo)

X = df_nvo_mm[['alcohol', 'volatile acidity','sulphates','chlorides']]
y = df_nvo['quality']


X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1), test_size=0.2, random_state=3)

lr_multiple = linear_model.LinearRegression()

lr_multiple.fit(X_train, y_train)

Y_pred_multiple = lr_multiple.predict(X_test)

print('Coeficients: ', lr_multiple.coef_)

print('\nIntercepts: ',lr_multiple.intercept_)

print("MinMax")
print('\nPrecision: ', lr_multiple.score(X_train, y_train))




#Normalizar
df_nvo_nrm= mean_norm(df_nvo)

X = df_nvo_nrm[['alcohol', 'volatile acidity','sulphates','chlorides']]
y = df_nvo['quality']


X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1), test_size=0.2, random_state=3)

lr_multiple = linear_model.LinearRegression()

lr_multiple.fit(X_train, y_train)

Y_pred_multiple = lr_multiple.predict(X_test)

print('Coeficients: ', lr_multiple.coef_)

print('\nIntercepts: ',lr_multiple.intercept_)

print("ZZZZ")
print('\nPrecision: ', lr_multiple.score(X_train, y_train))


#P-VALUE: tiene que ver con las colas, si es muy grande, no es bueno