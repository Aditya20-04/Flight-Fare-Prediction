import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



st.markdown("<h1 style='text-align:center;'>Flight Fare Prediction Machine Learning Project</h1>", unsafe_allow_html=True)

st.image("flight.jpg")

uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    # Read the uploaded CSV (not a fixed file)
    df = pd.read_csv("Clean_Dataset.csv.zip")
    
    st.write("Preview of uploaded data:")
    st.write(df.head())
    st.html('<br><br>')
    df=df.drop('Unnamed: 0',axis=1)
    st.write(df)
    st.html('<br><br>')
    st.write(df.tail())
    st.html("<br><br>")
    st.write(df.describe())
    st.html("<br><br>")
    df.isnull().sum()

    st.html("<br><br>")

    st.write(df.info())
    st.html('<br><br>')
    st.write(df.shape)
    st.html("<br><br>")
    st.write(df.size)
    st.html("<br><br>")
    fig, ax = plt.subplots()
    sns.scatterplot(x='duration', y='price', data=df, ax=ax)
    ax.set_title("Scatter Plot: Duration vs Price")
    ax.set_xlabel("Duration")
    ax.set_ylabel("Price")
    st.pyplot(fig)
    st.html("<br><br>")
    df1=df.groupby(['flight','airline'],as_index=False).count()
    st.write(df1.airline.value_counts())
    st.html("<br><br>")
    import warnings
    warnings.filterwarnings('ignore')
    fig = plt.figure(figsize=(8, 5))
    sns.countplot(df1['airline'],palette='hls')
    plt.title('Flights Count of Different Airlines',fontsize=15)
    plt.xlabel('Count',fontsize=15)
    plt.ylabel('Airline',fontsize=15)
    plt.show()
    st.pyplot(fig)
    st.html("<br><br>")
    df2=df.groupby(['airline','flight','class'],as_index=False).count()
    st.write(df2['class'].value_counts())
    st.html("<br><br>")
    fig=plt.figure(figsize=(10,5))
    df2['class'].value_counts().plot(kind='pie',autopct='%.2f',cmap="cool")
    plt.title("Classes of Diferent Airlines")

    # Show in Streamlit
    st.pyplot(fig)
    st.html("<br><br>")

    fig=plt.figure(figsize=(10,5))
    st.html("<br><br>")
    sns.boxplot(x='airline',y='price',data=df,palette='hls')
    plt.title("Boxplot Airline VS Price")
    plt.xlabel("Airline")
    plt.ylabel("Price")
    st.pyplot(fig)

    st.html("<br><br>")
    fig=plt.figure(figsize=(10,5))
    sns.boxplot(x='class',y='price',data=df,palette='hls')
    plt.title("BoxPlot Class VS price")
    plt.xlabel("Class")
    plt.ylabel("Price")
    st.pyplot(fig)

    st.html("<br><br>")
    fig=plt.figure(figsize=(10,5))
    sns.boxplot(x='stops',y='price',data=df,palette='hls')
    plt.title("Boxplot Stop Vs price")
    plt.xlabel("Stop")
    plt.ylabel("Price")
    st.pyplot(fig)

    st.html("<br><br>")
    fig=plt.figure(figsize=(10,5))
    sns.boxplot(x="departure_time",y='price',data=df,palette='hls')
    plt.title("Boxplot Departure Time VS Price")
    plt.xlabel("Departure Time")
    plt.ylabel("Price")
    st.pyplot(fig)

    st.html("<br><br>")
    sns.boxplot(x="arrival_time",y='price',data=df,palette='hls')
    plt.title("Boxplot Arrival Time VS Price")
    plt.xlabel("Arrival Time")
    plt.ylabel("Price")
    st.pyplot(fig)

    st.html("<br><br>")
    fig=plt.figure(figsize=(10,5))
    sns.boxplot(x="source_city",y='price',data=df,palette='hls')
    plt.title("Boxplot Source City VS Price")
    plt.xlabel("Source City")
    plt.ylabel("Price")
    st.pyplot(fig)

    st.html("<br><br>")
    fig=plt.figure(figsize=(10,5))
    sns.boxplot(x="destination_city",y='price',data=df,palette='hls')
    plt.title("Boxplot Destination City  VS Price")
    plt.xlabel("Destination City")
    plt.ylabel("Price")
    st.pyplot(fig)

    st.html("<br><br>")
    plt.style.use('dark_background')
    fig=plt.figure(figsize=(20,8))
    sns.lineplot(data=df,x='duration',y='price',hue='class',palette='hls')
    plt.title('Ticket Price Versus Flight Duration Based on Class')
    plt.xlabel('Duration')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig)

    st.html("<br><br>")
    plt.figure(figsize=(20,8))
    sns.lineplot(data=df,x='days_left',y='price',palette='hls')
    plt.title('Days Left for Departure Ticket Price',fontsize=20)
    plt.xlabel('Duration',fontsize=15)
    plt.ylabel('Price',fontsize=15)
    plt.show()
    st.pyplot(fig)

    st.html("<br><br>")
    fig=plt.figure(figsize=(20,8))
    sns.lineplot(data=df,x='days_left',y='price',palette='hls',hue='airline')
    plt.title('Days Left for Departure Ticket Price',fontsize=20)
    plt.xlabel('Duration',fontsize=15)
    plt.ylabel('Price',fontsize=15)
    plt.show()
    st.pyplot(fig)

    st.html("<br><br>")
    st.write(df.groupby(['airline','class','source_city','destination_city','flight'],as_index=False).count().groupby(['source_city','destination_city'],as_index=False)['flight'].count().head(10))
    
    st.html("<br><br>")
    st.write(df.groupby(['airline','source_city','destination_city'],as_index=False)['price'].mean().head(10))

    le=LabelEncoder()
    for col in df.columns:
        if df[col].dtype=='object':
            df[col]=le.fit_transform(df[col])
            
    
    st.html("<br><br>")
    #set features and target
    x=df.drop('price',axis=1)
    y=df['price']

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

    x_train.shape,x_test.shape,y_train.shape,y_test.shape

    st.html("<br><br>")
    mmscaler=MinMaxScaler(feature_range=(0,1))
    x_train=mmscaler.fit_transform(x_train)
    x_test=mmscaler.fit_transform(x_test)
    x_train=pd.DataFrame(x_train)
    x_test=pd.DataFrame(x_test)  

    st.html("<br><br>")
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    st.write(x_train)
    st.html("<br><br>")
    st.write(x_test)

    lr=LinearRegression()
    st.write(lr.fit(x_train,y_train))

    st.html("<br><br>")
    st.write("Coefficient or slope: ", lr.coef_)
    st.html("<br><br>")
    st.write("Intercept: ", lr.intercept_)

    st.html("<br><br>")
    y_pred=lr.predict(x_test)
    y_pred

    st.html("<br><br>")
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    st.write("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
    st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred))
    st.write("Root Mean squared Error:",rmse)

    st.html("<br><br>")
    st.write("Accuray R2 score",r2_score(y_test,y_pred))

    st.html("<br><br>")
    r_squared = round(metrics.r2_score(y_test, y_pred),6)
    adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),6)
    st.write('Adj R Square: ', adjusted_r_squared)

    st.html("<br><br>")
    dtr=DecisionTreeRegressor()
    st.write(dtr.fit(x_train,y_train))
    
    st.html("<br><br>")
    y_pred=dtr.predict(x_test)
    y_pred

    st.html("<br><br>")
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    st.write("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
    st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred))
    st.write("Root Mean squared Error:",rmse)

    st.html("<br><br>")
    st.write("Accuray R2 score",r2_score(y_test,y_pred))

    st.html("<br><br>")
    r_squared = round(metrics.r2_score(y_test, y_pred),6)
    adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),6)
    st.write('Adj R Square: ', adjusted_r_squared)


    st.html("<br><br>")
    rfg=RandomForestRegressor()
    st.write(rfg.fit(x_train,y_train))

    st.html("<br><br>")
    y_pred=rfg.predict(x_test)
    y_pred

    st.html("<br><br>")
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    st.write("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
    st.write("Mean Squared Error:",mean_squared_error(y_test,y_pred))
    st.write("Root Mean squared Error:",rmse)

    st.html("<br><br>")
    st.write("Accuray R2 score",r2_score(y_test,y_pred))

    st.html("<br><br>")
    r_squared = round(metrics.r2_score(y_test, y_pred),6)
    adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),6)
    st.write('Adj R Square: ', adjusted_r_squared)

    st.html("<br><br>")
    corr=df.corr()

    st.html("<br><br>")
    fig=plt.figure(figsize=(16,5))
    sns.heatmap(corr,annot=True,fmt='.2f',cmap='viridis')
    st.pyplot(fig)

    st.html("<br><br>")
    fig=plt.figure(figsize=(10,5))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.show()
    st.pyplot(fig)

    st.html("<br><br>")
    fig=plt.figure(figsize=(10,5))
    sns.lineplot(x=range(len(y_test)), y=y_test, label='Actual')
    sns.lineplot(x=range(len(y_pred)), y=y_pred, label='Predicted')
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Actual vs Predicted Over Time")
    plt.legend()
    plt.show()
    st.pyplot(fig)
