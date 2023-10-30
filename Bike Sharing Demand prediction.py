# Import required dependencies
import pandas as pd
pd.options.display.max_columns=999
import matplotlib.pyplot as plt, seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
import numpy as np       


# Load dataset
bike_dataset = pd.read_csv('hour.csv')
# Show dataset shape
bike_dataset.shape
# Show the first and last five samples in the dataset
bike_dataset.head()
bike_dataset.tail()
# Show some statistical info about the dataset
bike_dataset.describe()


# Check if there is any none(missing) values in the dataset to decide if will make a data cleaning or not
bike_dataset.isnull().sum()


# Rename some columns
print(bike_dataset.columns)
bike_dataset = bike_dataset.rename(columns={'yr':'year',
                                            'mnth':'month',
                                            'hum':'humidity',
                                            'weathersit':'weather',
                                            'cnt':'count',
                                            'hr':'hour'})
# Show dataset columns after rename some columns
print(bike_dataset.columns)


# Show unique values for each column
for col in bike_dataset.columns:
    print(f"the columns {col} has: ")
    print(len(bike_dataset[col].unique()),end="\n\n")

# Count values for each columns that has counted values and plot this count
plt.figure(figsize=(5,5))
for col in bike_dataset.columns:
    if col not in ['instant','dteday','count','registered','casual','windspeed','humidity','atemp','temp']:
        print(f"the column {col} has: ")
        print(bike_dataset[col].value_counts())
        sns.countplot(x=col,data=bike_dataset)
        plt.show()
        print()     
# Visualize distribtuion for each columns that has a lot of values
plt.figure(figsize=(5,5))
for col,color_dist in zip(['count','temp'],['red','blue']):
    print(f"the disttribution of column {col} is: ")
    sns.distplot(bike_dataset[col],color=color_dist)
    plt.show()

qqplot(bike_dataset['count'],line='s')
plt.title('Theoritical quantities')
plt.show()
# change the distribution of count column
bike_dataset['count'] = np.log(bike_dataset['count'])
# Plot the distribution after change it
sns.distplot(bike_dataset['count'],color='red')
plt.show()
qqplot(bike_dataset['count'],line='s')
plt.title('Theoritical quantities')
plt.show()



# Bivariate analysis for weekday and count columns
weekday_relation = bike_dataset.pivot_table(index='weekday',values='count',aggfunc=np.mean)
weekday_relation.plot(kind='bar',figsize=(10,10))
plt.xlabel('weekday')
plt.ylabel('count')
plt.title('weekday and count analysis')
plt.xticks(rotation=0)
plt.show()
# Bivariate analysis for weekday and count columns
holiday_relation = bike_dataset.pivot_table(index='holiday',values='count',aggfunc=np.mean)
holiday_relation.plot(kind='bar',figsize=(10,10))
plt.xlabel('holiday')
plt.ylabel('count')
plt.title('holiday and count analysis')
plt.xticks(rotation=0)
plt.show()
# Bivariate analysis for season and count columns
holiday_relation = bike_dataset.pivot_table(index='season',values='count',aggfunc=np.mean)
holiday_relation.plot(kind='bar',figsize=(10,10))
plt.xlabel('season')
plt.ylabel('count')
plt.title('season and count analysis')
plt.xticks(rotation=0)
plt.show()
# Bivariate analysis for month and count columns
holiday_relation = bike_dataset.pivot_table(index='month',values='count',aggfunc=np.mean)
holiday_relation.plot(kind='bar',figsize=(10,10))
plt.xlabel('month')
plt.ylabel('count')
plt.title('month and count analysis')
plt.xticks(rotation=0)
plt.show()

# plot bars of count for hour
plt.figure(figsize=(5,5))
sns.barplot(data=bike_dataset,x='hour',y='count',estimator=sum/np.mean)
plt.show()
# plot bars of count for month
plt.figure(figsize=(5,5))
sns.barplot(data=bike_dataset,x='month',y='count')
plt.show()
# plot bars of count for season
plt.figure(figsize=(5,5))
sns.barplot(data=bike_dataset,x='season',y='count')
plt.show()
# plot bars of count for year
plt.figure(figsize=(5,5))
sns.barplot(data=bike_dataset,x='year',y='count')
plt.show()

# plot relation between humidity and count columns
sns.regplot(x=bike_dataset['humidity'],y=bike_dataset['count'])
plt.title('relation between humidity and count')
plt.show()
# plot relation between temp and count columns
sns.regplot(x=bike_dataset['temp'],y=bike_dataset['count'])
plt.title('relation between humidity and count')
plt.show()




# plot count during weekends and weekdays
plt.figure(figsize=(10,10))
sns.pointplot(data=bike_dataset, x='hour', y='count', hue='weekday')
plt.title('count during weekends and weekdays')
plt.show()
# plot casual during weekends and weekdays
sns.pointplot(data=bike_dataset, x='hour', y='casual', hue='weekday')
plt.title('casual during weekends and weekdays')
plt.show()
# plot registered during weekends and weekdays
sns.pointplot(data=bike_dataset, x='hour', y='registered', hue='weekday')
plt.title('registered during weekends and weekdays')
plt.show()
# plot count during weather
sns.pointplot(data=bike_dataset, x='hour', y='count', hue='weather')
plt.title('count during weather')
plt.show()
# plot count during season
sns.pointplot(data=bike_dataset, x='hour', y='count', hue='season')
plt.title('count during season')
plt.show()


# Remove some columns as instant, dteday and year columns
bike_dataset = bike_dataset.drop(columns=['instant','dteday','year'],axis=1)

# Find correlation between all dataset features
correlation_values = bike_dataset.corr()
# plot correlation 
plt.figure(figsize=(15,15))
sns.heatmap(correlation_values,cbar=True,square=True,annot=True,annot_kws={'size':8},cmap='Blues')
plt.show() 


# Convert int columns into categorical columns
cols = ['season','month','hour','holiday','weekday','workingday','weather']
for col in cols:
    bike_dataset[col] = bike_dataset[col].astype('category')

bike_dataset.head()
bike_dataset_oh = bike_dataset
bike_dataset_oh.head()
# Make one hot encoder for the columns in cols list
def one_hot_encoder(data,column):
    data = pd.concat([data,pd.get_dummies(bike_dataset[column],prefix=column,drop_first=True)],axis=1)
    data = data.drop(columns = [column],axis=1)
    return data
for col in cols:
    bike_dataset_oh = one_hot_encoder(bike_dataset_oh,col)
# Show the bike dataset after make one hot encoder to it
bike_dataset_oh.head()



# Split data into input and label data
X = bike_dataset_oh.drop(columns=['atemp','windspeed','casual','registered','count'],axis=1)
Y = bike_dataset_oh['count']
# Split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)


def plot_train_test(y_train,predicted_y_train,y_test,predicted_y_test):
    plt.figure(figsize=(7,7))
    plt.scatter(y_train,predicted_y_train,color='red',marker='X')
    plt.plot(range(np.max(y_train)),color='black')
    plt.title('show actual and predicted train values')
    plt.xlabel('actual values')
    plt.ylabel('predicted values')
    plt.show()
    plt.scatter(y_test,predicted_y_test,color='blue',marker='o')
    plt.plot(range(np.max(y_test)),color='black')
    plt.title('show actual and predicted test values')
    plt.xlabel('actual values')
    plt.ylabel('predicted values')
    plt.show()

# Train model and test it
def Train_predict(model,x_train,x_test,y_train,y_test):
    # Train model 
    model.fit(x_train,y_train)
    # Make the model predict on train and test input data
    predicted_train_data = model.predict(x_train)
    predicted_test_data = model.predict(x_test)
    # plot difference between predicted and actual values
    plot_train_test(y_train,predicted_train_data,y_test,predicted_test_data)
    # Evaluate model
    accuracy_train_pred = r2_score(y_train,predicted_train_data)
    accuracy_test_pred = r2_score(y_test,predicted_test_data)
    # Return accuracy
    return [accuracy_train_pred,accuracy_test_pred]

# Create Model to Find its accuracy and compare between it
Models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    Ridge(),
    HuberRegressor(),
    ElasticNetCV(),
    GradientBoostingRegressor(),
    ExtraTreesRegressor()
]
# Train each model
for model in Models: 
    print("the model:",model,"has")
    accuracy_train_pred, accuracy_test_pred = Train_predict(model,x_train,x_test,y_train,y_test)
    print("\t",accuracy_train_pred,"on train data",accuracy_test_pred,"on test data")



