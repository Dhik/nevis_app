from django.shortcuts import render
from .forms import UserForm,UserProfileInfoForm
from django.http import JsonResponse

# Extra Imports for the Login and Logout Capabilities
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.contrib.auth.decorators import login_required
import pandas as pd
from django.conf import settings
import os

#predict analysis
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .models import News

import requests
import json

#predict analysis
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


#summary
from langchain_openai import OpenAI
from langchain import verbose
import langchain.globals
from dotenv import load_dotenv
from datetime import datetime

#Random Regressor
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

load_dotenv()
# api_key = os.getenv('OPENAI_API_KEY')
# llm= OpenAI(api_key= api_key, temperature=0.9)


def pred_ranres(request):
    api_url = 'http://127.0.0.1:8000/nevisapp/data_aluminium/'  # Replace this with your API endpoint
    
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            api_data = response.json()
            
            # Convert API data to DataFrame
            df = pd.DataFrame(api_data['results'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Split data into features (X) and target variable (y)
            X = df[['Copper Price', 'BADI -2', 'MJP', 'USD Index', 'AL Prod (Global)', 'AL Cons (Global)', 'Aluminium stocks', 'Alumina Index', 'Oil Prices']]
            y = df['Aluminium Price']
            
            # Initialize RandomForestRegressor model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the parameters
            
            # Fit the model
            rf_model.fit(X, y)
            
            # Forecasting future values for 5 months using ExponentialSmoothing
            forecasted_dates = pd.date_range(start=df['Date'].max(), periods=6, freq='M')[1:]  # 5 months forward
            forecasted_features = pd.DataFrame(index=forecasted_dates, columns=X.columns)

            for feature in X.columns:
                model = ExponentialSmoothing(df[feature], seasonal='additive', seasonal_periods=12).fit()
                forecasted_features[feature] = model.forecast(5)

            # Make predictions using the RandomForestRegressor model
            predictions = rf_model.predict(forecasted_features)
            
            # Calculate metrics
            mae = mean_absolute_error(y, rf_model.predict(X))
            mse = mean_squared_error(y, rf_model.predict(X))
            rmse = np.sqrt(mse)
            r2 = r2_score(y, rf_model.predict(X))
            
            # Create DataFrame for predicted and actual prices
            results_df = pd.DataFrame({'Date': df['Date'], 'actual_price': y, 'predictive_price': rf_model.predict(X)})
            
            # Convert 'Date' column to string
            results_df['Date'] = results_df['Date'].dt.strftime('%Y-%m-%d')
            results_df.rename(columns={'Date': 'date'}, inplace=True)
            
            # Convert DataFrame to JSON
            results_json = results_df.to_json(orient='records', double_precision=15)

            # Return the JSON response with column names and metrics
            response_data = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "data": json.loads(results_json),
                "future_predictions": list(predictions)
            }
            return JsonResponse(response_data)
        
        else:
            return JsonResponse({'error': 'Failed to fetch data from API'}, status=response.status_code)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

alum= '15000 $/ton'
tanggal_dt= '2030-26-12'
excel_file_path = os.path.join(settings.BASE_DIR, 'dataset', 'test.xlsx')

df = pd.read_excel(excel_file_path)
df['Date'] = pd.to_datetime(df['Date'])

data = df[['Date', 'Aluminium Price']].rename(columns={'Date': 'ds', 'Aluminium Price': 'y'})
data.set_index('ds', inplace=True)

model = ExponentialSmoothing(data, seasonal='add', seasonal_periods=12)
fitted_model = model.fit()
predictions = fitted_model.forecast(steps=10)
df_pred = pd.DataFrame(list(predictions.items()), columns=['Date', 'Value'])
merged_df = pd.merge(df[['Date','Aluminium Price']], df_pred, on='Date', how='outer')
merged_df['Aluminium Price'] = merged_df['Aluminium Price'].fillna(merged_df['Value'])
merged_df.drop(columns=['Value'], inplace=True)
merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
merged_df.rename(columns={'Date': 'date', 'Aluminium Price': 'aluminium_price'}, inplace=True)


def get_pred_al_dl(request):
    # Change 'your_excel_file.xlsx' to the path of your Excel file
    #model_file_path = 'model/LSTM-07.h5'
    excel_file_path = os.path.join(settings.BASE_DIR, 'dataset', 'test.xlsx')
    #model_file_path = os.path.join(settings.BASE_DIR, 'model', 'LSTM-07.h5')

    # Read data from Excel file using pandas
    df = pd.read_excel(excel_file_path)
    #model = tf.keras.models.load_model(model_file_path, compile=False)
    # Convert 'Date' column to datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[['Date', 'Aluminium Price']]
    df['Date'] = pd.to_datetime(df['Date'])

    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    missing_dates = date_range[~date_range.isin(df['Date'])]

    # =========================================================
    df_eda = df.copy()

    df_eda.set_index('Date', inplace=True)
    df_eda = df_eda.sort_index(ascending=True)
    min_date = df_eda.index.min()
    max_date = df_eda.index.max()
    freq = 'D'

    index_lengkap = pd.date_range(start=min_date, end=max_date, freq='D')
    df_eda = df_eda.reindex(index_lengkap)

    # =====================================================
    df_lin = df_eda.copy()
    columns_to_impute = df_lin.columns

    for column in columns_to_impute:
    # Fit and transform the column
        df_lin[column] = df_lin[[column]].interpolate(method='cubic', axis=0)

    # =======================================================
    df = df_lin.copy()
    train_data, test_data = df[0:int(len(df)*0.85)], df[int(len(df)*0.85):]

    train = train_data.iloc[:,0:1]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)

    timesteps = 100
# timesteps = 50 # percobaan 1

    X_train = []
    y_train = []
    for i in range(timesteps, train.shape[0]):
        X_train.append(train_scaled[i-timesteps:i, 0])
        y_train.append(train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #Long short-term memory(LSTM) cells are used in recurrent neural networks that learn to predict the future from sequences of variable lengths.
    model = Sequential()
    #Adding the first LSTM layer
    #return_sequences=True means whether to return the last output in the output sequence. It basically tells us that there is another LSTM layer ahead in the network.
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    #Dropout regularisation for tackling overfitting
    model.add(Dropout(0.20))
    model.add(LSTM(units = 100, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    #Adding the output layer
    model.add(Dense(units = 1))
    #Compiling the model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    #Fitting the model to the Training set
    history = model.fit(X_train, y_train, epochs = 2, batch_size = 32)

    real_new_cases = test_data.iloc[:,0:1].values #

    combine = pd.concat((train_data['Aluminium Price'], test_data['Aluminium Price']), axis = 0)
    test_inputs = combine[len(combine) - len(test_data) - timesteps:].values
    test_inputs = test_inputs.reshape(-1,1)
    test_inputs = scaler.transform(test_inputs)


    X_test = []
    for i in range(timesteps, test_data.shape[0]+timesteps):
        X_test.append(test_inputs[i-timesteps:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted = model.predict(X_test)
    #inverse_transform because prediction is done on scaled inputs
    predicted = scaler.inverse_transform(predicted)

    # .............calculate metrics...............................
    mae = mean_absolute_error(real_new_cases, predicted)
    mse = mean_squared_error(real_new_cases, predicted)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(real_new_cases, predicted)
    r2 = r2_score(real_new_cases, predicted)

    # Create a dictionary to hold the error metrics
    lstm_error = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R-squared': r2
    }

    # Convert the dictionary to JSON
    lstm_error_json = json.dumps(lstm_error, indent=4)

    # >>>>>>>>>>>>>>>>>>  Forcast >>>>>>>>>>>>>>>>>>>>>>>

    timestep = 100
    def insert_end(Xin,new_input):
        for i in range(timestep-1):
            Xin[:,i,:] = Xin[:,i+1,:]
        Xin[:,timestep-1,:] = new_input
        return Xin
    


    future = 365
    forcast = []
    Xin = X_test[-1:]
    for i in range(future):
        out = model.predict(Xin, batch_size=1, verbose=0)
        forcast.append(out[0,0])
        Xin = insert_end(Xin,out[0,0])


    forcasted_output = np.asanyarray(forcast)
    forcasted_output = forcasted_output.reshape(-1,1)
    forcasted_output = scaler.inverse_transform(forcasted_output)

    forcasted_output = pd.DataFrame(forcasted_output)
    date = pd.DataFrame(pd.date_range(start='2024-01-02',periods=365,freq='D'))
    df_result = pd.concat([date,forcasted_output],axis=1)
    df_result.columns = "Date","Forecasted"


    # -------- handle result -----------------------------
    df_forcast = df_result.copy()
    test = test_data.iloc[:,0:1]
    test.index.name = 'Date'

    train.index.name = 'Date'


    df_forcast = df_forcast.rename(columns={'Forecasted': 'Aluminium Price'})
    df_forcast = df_forcast.dropna(subset=['Aluminium Price'])
    df_forcast.set_index('Date', inplace=True)


    concatenated_df = pd.concat([train, test, df_forcast])

    merged_df = concatenated_df.copy()
    merged_df.reset_index(inplace=True)
    merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
    #merged_df.rename(columns={'Date': 'date', 'Aluminium Price': 'aluminium_price'}, inplace=True)
    merged_df.rename(columns={'Aluminium Price': 'aluminium_price'}, inplace=True)
    merged_df.rename(columns={'Date': 'date'}, inplace=True)


    # Convert 'Date' column to string
    #merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
    #merged_df.rename(columns={'Date': 'date', 'Aluminium Price': 'aluminium_price'}, inplace=True)

    merged_df_json = merged_df.to_json(orient='records', double_precision=15)

    # Return the JSON response with column names
    response_data = {"data": json.loads(merged_df_json), 'metrics': json.loads(lstm_error_json)}
    #response_data = {"data": json.loads(merged_df_json)}
    return JsonResponse(response_data)


def req_conclusion(request):
    prompt_conclusion = f'''Dengan menggunakan bahasa indonesia, berikan 1 paragraf kesimpulan dari nilai aluminium {predictions} , apakah mengalami 
    penurunan atau kenaikan sesuaikan dengan pola pada{merged_df} dan berikan faktor serta kondisi harga aluminium dunia saat ini'''
    conclusion= llm.invoke(prompt_conclusion)
    conclusion_json= json.dumps(conclusion)
    return JsonResponse({'Conclusion:': conclusion_json})


def get_recommendation(request):
    prompt_recommendation = f'''Dengan menggunakan bahasa indonesia, berikan 3 bullet point rekomendasi untuk 
    stakeholder yang ingin membeli aluminium berdasarkan dari {predictions}, sesuaikan dengan data {merged_df}'''
    recommendation= llm.invoke(prompt_recommendation)
    recommendation_json= json.dumps(recommendation)
    return JsonResponse({'Recommedation:': recommendation_json})

def coba(request):
    return JsonResponse({'test': "hallo"})

def pred_logres(request):
    api_url = 'http://127.0.0.1:8000/nevisapp/data_aluminium/'  # Replace this with your API endpoint
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            api_data = response.json()
            
            # Convert API data to DataFrame
            df = pd.DataFrame(api_data['results'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Split data into features (X) and target variable (y)
            X = df[['Copper Price', 'BADI -2', 'MJP', 'USD Index', 'AL Prod (Global)', 'AL Cons (Global)', 'Aluminium stocks', 'Alumina Index', 'Oil Prices']]
            y = df['Aluminium Price']
            
            # Fit Lasso regression model
            lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha value
            lasso_model.fit(X, y)
            
            # Make predictions
            predictions = lasso_model.predict(X)
            
            # Calculate metrics
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            
            # Create DataFrame for predicted and actual prices
            results_df = pd.DataFrame({'Date': df['Date'], 'actual_price': y, 'predictive_price': predictions})
            
            # Convert 'Date' column to string
            results_df['Date'] = results_df['Date'].dt.strftime('%Y-%m-%d')
            results_df.rename(columns={'Date': 'date'}, inplace=True)
            
            # Convert DataFrame to JSON
            results_json = results_df.to_json(orient='records', double_precision=15)

            # Return the JSON response with column names and metrics
            response_data = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "data": json.loads(results_json),
                
            }
            return JsonResponse(response_data)
        
        else:
            return JsonResponse({'error': 'Failed to fetch data from API'}, status=response.status_code)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def import_data(request):
    # Change 'your_excel_file.xlsx' to the path of your Excel file
    excel_file_path = os.path.join(settings.BASE_DIR, 'dataset', 'test.xlsx')

    # Read data from Excel file using pandas
    df = pd.read_excel(excel_file_path)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Convert dataframe to JSON
    data_json = df.to_json(orient='records')
    # Remove leading and trailing double quotes
    data_string = data_json.strip('"')

    # Convert to JSON
    data_json = json.loads(data_string)

    # Return the JSON response
    return JsonResponse({"results": data_json})

def al_predict(request):
    return render(request, 'nevisapp/merged_df.html')

def get_pred_al(request):
    api_url = 'http://127.0.0.1:8000/nevisapp/data_aluminium/'  # Replace this with your API endpoint
    
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            api_data = response.json()
            
            # Convert API data to DataFrame
            df = pd.DataFrame(api_data['results'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Perform time series forecasting
            data = df[['Date', 'Aluminium Price']].rename(columns={'Date': 'ds', 'Aluminium Price': 'y'})
            data.set_index('ds', inplace=True)

            model = ExponentialSmoothing(data, seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
            predictions = fitted_model.forecast(steps=5)
            df_pred = pd.DataFrame(list(predictions.items()), columns=['Date', 'Value'])
            merged_df = pd.merge(df[['Date','Aluminium Price']], df_pred, on='Date', how='outer')
            merged_df['Aluminium Price'] = merged_df['Aluminium Price'].fillna(merged_df['Value'])
            merged_df.drop(columns=['Value'], inplace=True)

            # Convert 'Date' column to string
            merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
            merged_df.rename(columns={'Date': 'date', 'Aluminium Price': 'aluminium_price'}, inplace=True)

            merged_df_json = merged_df.to_json(orient='records', double_precision=15)

            # Return the JSON response with column names
            response_data = {"data": json.loads(merged_df_json)}
            return JsonResponse(response_data)
        
        else:
            return JsonResponse({'error': 'Failed to fetch data from API'}, status=response.status_code)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def al_predict(request):
    return render(request, 'nevisapp/merged_df.html')


def import_and_return_json(request):
    # Change 'your_excel_file.xlsx' to the path of your Excel file
    excel_file_path = os.path.join(settings.BASE_DIR, 'dataset', 'test.xlsx')

    # Read data from Excel file using pandas
    df = pd.read_excel(excel_file_path)

    # # Convert dataframe to JSON
    # data_json = df.to_json(orient='records')
    # # Remove leading and trailing double quotes
    # data_string = data_json.strip('"')

    # # Convert to JSON
    # data_json = json.loads(data_string)

    # # Return the JSON response
    # return JsonResponse({"results": data_json})

    # Convert the JSON response to a DataFrame
    # data = {
    #     "Date": [1293840000000, 1296518400000, 1298937600000, 1301616000000],
    #     "Aluminium Price": [2439.7, 2515.26, 2555.5, 2667.4166666667],
    #     "Copper Price": [9533.2, 9884.9, 9503.3586956522, 9482.75],
    #     "BADI -2": [2321.3181818182, 2030.9444444444, 1401.8, 1181.1],
    #     "MJP": [112.5, 112.5, 113, 114],
    #     "USD Index": [79.1553809524, 77.7776, 76.2876521739, 74.6897619048],
    #     "AL Prod (Global)": [3758.482821, 3416.918966, 3788.014108, 3800.103276],
    #     "AL Cons (Global)": [3646.5591542059, 3210.0681146571, 3674.2978401862, 3823.5741200196],
    #     "Aluminium stocks": [11886.3473095413, 12093.1981608842, 12206.9144286979, 12183.4435846784],
    #     "Alumina Index": [378.2975, 392.4375, 402.1475, 410.37],
    #     "Oil Prices": [97, 104, 115, 123]
    # }

    # df = pd.DataFrame(data)

    # Define features and target
    X = df.drop(columns=["Aluminium Price"])
    y = df["Aluminium Price"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = np.mean((predictions - y_test) ** 2)
    return JsonResponse({"mse": mse})


# Create your views here.
def index(request):
    return render(request,'nevisapp/index.html')

def dashboard(request):
    return render(request,'nevisapp/index.html')

def get_news(request):
    data = News.objects.all()
    data = {"results": data}
    return render(request, 'nevisapp/news.html', data)

def get_news_by_id(request, keywords):
    try:
        data = News.objects.filter(keywords=keywords)
        serialized_data = []
        for news in data:
            serialized_data.append({
                "id": news.id,
                "title": news.title,
                "source": news.source,
            })
        return JsonResponse(serialized_data, safe=False)
    except News.DoesNotExist:
        return JsonResponse({"error": "News not found"}, status=404)

def news_analysis(request):
    if request.method == 'POST':
        # First get the username and password supplied
        query = request.POST.get('query')

        # Set up the query parameters
        api_key = "e2a30aecb963d2caeb1fd6303c6c45ed641ec2df5a87afca772e7c0e63c37df7"
        engine = "google"

        # Construct the API URL
        url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}&engine={engine}"

        # Send a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # List to hold all the results
            results = []

            for dt in data['organic_results']:
                link = dt['link']
                # Check if a News instance with the same link already exists
                if not News.objects.filter(link=link).exists():
                    result = News()
                    result.title = dt['title']
                    result.link = link
                    result.redirect_link = dt['redirect_link']
                    result.displayed_link = dt['displayed_link']
                    result.favicon = dt['favicon']
                    result.snippet = dt['snippet']
                    result.snippet_highlighted_words = ''
                    result.source = dt['source']
                    result.save()
                    results.append(result)
            
            data = {"results": results}
            # Render the template with the results
            return render(request, 'nevisapp/news_analysis.html', data)
        else:
            # Handle the error
            print(f"Error: {response.status_code}")

    return render(request, 'nevisapp/news_analysis.html')
    

@login_required
def special(request):
    # Remember to also set login url in settings.py!
    # LOGIN_URL = '/basic_app/user_login/'
    return HttpResponse("You are logged in. Nice!")

@login_required
def user_logout(request):
    # Log out the user.
    logout(request)
    # Return to homepage.
    return HttpResponseRedirect(reverse('index'))

def register(request):

    registered = False

    if request.method == 'POST':

        # Get info from "both" forms
        # It appears as one form to the user on the .html page
        user_form = UserForm(data=request.POST)
        profile_form = UserProfileInfoForm(data=request.POST)

        # Check to see both forms are valid
        if user_form.is_valid() and profile_form.is_valid():

            # Save User Form to Database
            user = user_form.save()

            # Hash the password
            user.set_password(user.password)

            # Update with Hashed password
            user.save()

            # Now we deal with the extra info!

            # Can't commit yet because we still need to manipulate
            profile = profile_form.save(commit=False)

            # Set One to One relationship between
            # UserForm and UserProfileInfoForm
            profile.user = user

            # Check if they provided a profile picture
            if 'profile_pic' in request.FILES:
                print('found it')
                # If yes, then grab it from the POST form reply
                profile.profile_pic = request.FILES['profile_pic']

            # Now save model
            profile.save()

            # Registration Successful!
            registered = True

        else:
            # One of the forms was invalid if this else gets called.
            print(user_form.errors,profile_form.errors)

    else:
        # Was not an HTTP post so we just render the forms as blank.
        user_form = UserForm()
        profile_form = UserProfileInfoForm()

    # This is the render and context dictionary to feed
    # back to the registration.html file page.
    return render(request,'nevisapp/registration.html',
                          {'user_form':user_form,
                           'profile_form':profile_form,
                           'registered':registered})

def user_login(request):

    if request.method == 'POST':
        # First get the username and password supplied
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Django's built-in authentication function:
        user = authenticate(username=username, password=password)

        # If we have a user
        if user:
            #Check it the account is active
            if user.is_active:
                # Log the user in.
                login(request,user)
                # Send the user back to some page.
                # In this case their homepage.
                return HttpResponseRedirect(reverse('index'))
            else:
                # If account is not active:
                return HttpResponse("Your account is not active.")
        else:
            print("Someone tried to login and failed.")
            print("They used username: {} and password: {}".format(username,password))
            return HttpResponse("Invalid login details supplied.")

    else:
        #Nothing has been provided for username or password.
        return render(request, 'nevisapp/login.html', {})


def get_analyze(request):

    if request.method == 'POST':
        # First get the username and password supplied
        query = request.POST.get('query')

        # Set up the query parameters
        api_key = "e2a30aecb963d2caeb1fd6303c6c45ed641ec2df5a87afca772e7c0e63c37df7"
        engine = "google"

        # Construct the API URL
        url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}&engine={engine}"

        # Send a GET request to the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            for dt in data['organic_results']:
                result = News()

                # Set the attributes of the model instance
                result.title = dt['title']
                result.link = dt['url']
                result.redirect_link = dt['redirect_link']
                result.displayed_link = dt['displayed_link']
                result.favicon = dt['favicon']
                result.snippet = dt['snippet']
                result.snippet_highlighted_words = dt['snippet_highlighted_words']
                result.source = dt['source']

                # Save the model instance to the database
                result.save()
        else:
            # Handle the error
            print(f"Error: {response.status_code}")

