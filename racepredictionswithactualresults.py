import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def calculate_sentiment(text):
    if text == 'D':
        return 1
    elif text == 'T':
        return 0
    else:
        return 0

# Load the race data
df = pd.read_csv('c:\\races\\penraces2022.csv', encoding='utf-8-sig', low_memory=False)

# Add a copy of the 'Name' column
df['HorseName'] = df['Name']

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort data by date and by horse
df.sort_values(['Name', 'Date'], inplace=True)

# Calculate the average finish position for each horse in the past three races
df['PastPerformance'] = df.groupby('Name')['ActualFinish'].transform(lambda x: x.rolling(30, min_periods=1).mean())

# Calculate the number of days since the horse's last race
df['DaysSinceLastRace'] = df.groupby('Name')['Date'].diff().dt.days

# Sort data by date and by jockey/trainer
df.sort_values(['Jockey', 'Date'], inplace=True)

# Calculate the average performance of the jockey in the past three races
df['JockeyPerformance'] = df.groupby('Jockey')['ActualFinish'].transform(lambda x: x.rolling(30, min_periods=1).mean())

# Add a new feature to denote if a horse was scratched
df['Scratched'] = df['ActualFinish'].apply(lambda x: 1 if x == 0 else 0)

#df.sort_values(['Trainer', 'Date'], inplace=True)

# Calculate the average performance of the trainer in the past three races
#df['TrainerPerformance'] = df.groupby('Trainer')['ActualFinish'].transform(lambda x: x.rolling(30, min_periods=1).mean())

# Convert column 21 to a uniform type (string as an example)
df.iloc[:, 21] = df.iloc[:, 21].astype(str)


# Perform one-hot encoding for 'Name' column
df = pd.get_dummies(df, columns=['Name'])

# Define columns to ignore
ignore_columns = [
    'RecID', 'Date', 'Name', 'RaceNumber', 'Post', 'PPCt', 'NLA',
    'LastBrisSpdRtg', 'BestBrisSpdRtg', 
    'AverageBrisSpdRtg', 'MedianBrisSpdRtg', 'BestLifeSpd', 'BestYearSpd', 'BestTrkSpd', 'WagerPost', 
    'Spd2F', 'Spd4F', 'SpdLate', 'Purse', 
    'LastFinishPos', 
    'LastBtnLns', 'LastDist', 'LastPurse', 'Finish2ndLastPos', 'Last2ndBtnLns', 'Last2ndDist', 'Last2ndPurse', 
    'FldSz', 'FinishPos', 'BtnLns', 'AvgLast3', 'Exacta', 'TriFecta', 'NLANorm', 'WgtNorm', 
    'LastSpdRtgNorm', 'BestSpdRtgNorm', 'AverageSpdRtgNorm', 'MedianSpdRtgNorm', 'LastBrisSpdRtgNorm', 
    'BestBrisSpdRtgNorm', 'AverageBrisSpdRtgNorm', 'MedianBrisSpdRtgNorm', 'BestLifeSpdNorm', 'BestYearSpdNorm', 
    'BestTrkSpdNorm', 'WagerPostNorm', 'Spd2FNorm', 'Spd4FNorm', 'SpdLateNorm', 'MLNorm', 'FinishPosNorm', 
    'LowestSpdRtg', 'LastFldSz', 'SecondFldSz', 'ThirdFldSz', 'ThirdFinPos', 'FourthFldSz', 'FourthFinPos', 
    'ThirdPurse', 'FourthPurse', 'LastRaceDate', 'Last2ndRaceDate', 'Checked', 'Normalized', 'WinnersTime',
    'TStarts', 'TWins', 'TPlaces', 'TShows', 'Trainer', 'TrainerNorm', 'JockeyNorm', 'ActualTime', 'MLRank',
    'ActualBtnLns', 'ActualOdds', 'ActualPurse', 'Starts', 'Wins', 'Places', 'Shows',
    'JStarts', 'JWins', 'JPlaces', 'JShows', 'Jockey'
]

# Define text fields
text_fields = ['Surface']

# Calculate sentiment and replace the original columns with sentiment values
for field in text_fields:
    sentiment_field = field + '_sentiment'
    if sentiment_field not in df.columns:
        df[sentiment_field] = df[field].apply(calculate_sentiment)
    df[field] = df[sentiment_field]

# Identify numeric columns to include in training
numeric_cols = df.select_dtypes(include=['int', 'float']).columns
numeric_cols = [col for col in numeric_cols if col not in ignore_columns]
numeric_cols = [col for col in numeric_cols if col not in text_fields]

# Add Scratched feature to numeric_cols
#numeric_cols.append('Scratched')

# print(numeric_cols)

# Replace infinities with NaN
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

# Calculate mean and check if all numeric_cols are included in the mean series
mean_values = df[numeric_cols].mean()
missing_cols = [col for col in numeric_cols if col not in mean_values.index]

if missing_cols:
    print(f"Mean calculation failed for the following columns: {missing_cols}")

# If no missing columns, proceed to fill NaN with mean
if not missing_cols:
    df[numeric_cols] = df[numeric_cols].fillna(mean_values)
else:
    print("Please check the data in the above columns.")


train_df = df[df['Scratched'] == 0]

# Create input features (X) and target variable (y)
X = train_df[numeric_cols]
y = train_df['ActualFinish']


# Check for any remaining NaNs or infinities
assert not X.isnull().values.any(), "There are still NaNs in the input data."
assert np.isfinite(X.to_numpy()).all(), "There are still infinities in the input data."

# Normalize target variable using StandardScaler
#scaler = StandardScaler()
#y = scaler.fit_transform(y.values.reshape(-1, 1))

# Normalize target variable using RobustScaler
#scaler = RobustScaler()
#y = scaler.fit_transform(y.values.reshape(-1, 1))

# Normalize target variable using MinMaxScaler
#scaler = MinMaxScaler()

#y = scaler.fit_transform(y.values.reshape(-1, 1))
# Normalize input features and target variable using RobustScaler
scaler = RobustScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))


# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Convert data types to float32
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.float32)
X = X.astype(np.float32)
y = y.astype(np.float32)

# Build the model architecture
model = Sequential()
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))  # added relu activation here

# Train the model on training data and validate on validation data
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=2048)
# model.fit(X, y, epochs=30, batch_size=2048)
# Load new race data
new_race = pd.read_csv('c:\\races\\race1testdata.csv', encoding='utf-8-sig', low_memory=False)
# and do the same for your new_race DataFrame
new_race['HorseName'] = new_race['Name']
new_race['ActualFinish'].fillna(-1, inplace=True)
new_race['ActualTime'].fillna(-1, inplace=True)
new_race['LastRaceDate'].fillna(0, inplace=True)
new_race['Last2ndRaceDate'].fillna(0, inplace=True)
new_race['Trainer'].fillna(0, inplace=True)
new_race['Jockey'].fillna(0, inplace=True)

# Perform one-hot encoding for 'Name' column in new race data
new_race = pd.get_dummies(new_race, columns=['Name'])

# Preprocess the new race data
for field in text_fields:
    sentiment_field = field + '_sentiment'
    if sentiment_field not in new_race.columns:
        new_race[sentiment_field] = new_race[field].apply(calculate_sentiment)

missing_cols = [col for col in numeric_cols if col not in new_race.columns]
if missing_cols:
    print("Warning: The following required columns are missing from the new_race data:", missing_cols)
    print("Adding missing columns with a default value of 0.")
    for col in missing_cols:
        new_race[col] = 0

new_race[numeric_cols] = new_race[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(new_race[numeric_cols].mean())
new_race = new_race.dropna(subset=['RecID', 'Date', 'RaceNumber', 'Surface', 'ActualTime', 'LastRaceDate', 'Last2ndRaceDate', 'Trainer', 'Jockey',
            'Checked', 'Normalized', 'WinnersTime', 'ActualTime', 'MLRank', 'JockeyNorm', 'TrainerNorm', 'ActualBtnLns', 'ActualOdds', 'ActualPurse',
            'FldSz', 'FinishPos', 'BtnLns', 'AvgLast3', 'Exacta', 'TriFecta', 'WagerPostNorm', 'FinishPosNorm'

 ])

# Make predictions on the new race data
predictions = model.predict(new_race[numeric_cols].astype(np.float32))

#print(predictions)

# Add the predicted positions to the new race data
new_race['PredictedFPos'] = predictions
new_race = new_race.sort_values('PredictedFPos')
#print(new_race.isnull().sum())
#print(new_race[numeric_cols].isna().any())
#print(new_race[numeric_cols].applymap(np.isinf).any())
new_race = new_race.sort_values(['Date', 'RaceNumber'])

# Print the order of finish
print("Order of finish:")
horse_name_cols = [col for col in new_race.columns if col.startswith('Name_')]
sorted_horse_name_cols = sorted(horse_name_cols, key=lambda x: new_race[x].iloc[0], reverse=True)
#for i, col in enumerate(sorted_horse_name_cols):
#    horse_name = col.split('_', 1)[1]
#    print(f"Position {i+1}: {horse_name}")

for i, row in new_race.iterrows():
    print(f"{row['Date']}\t{int(row['RaceNumber'])}\t{int(row['Post'])}\t{row['HorseName']}\t{round(row['PredictedFPos'], 2)}\t{row['ActualFinish']}\t${row['Earnings']}\t{row['ActualOdds']}\t{row['PPCt']}")



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()