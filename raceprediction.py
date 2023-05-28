import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from textblob import TextBlob

def calculate_sentiment(text):
    if text == 'D':
        return 1  # Preference for Dirt
    elif text == 'T':
        return 0  # Preference for Turf
    else:
        return 0  # Default to Turf preference

# Load data
df = pd.read_csv('c:\\races\\penraces2022.csv', encoding='utf-8-sig')

# One-hot encoding
df = pd.get_dummies(df, columns=['Name'])

# Define columns to ignore
ignore_columns = [
    'RecID', 'Name', 'Date', 'RaceNumber', 'Post', 'PPCt', 'NLA',  
    'LastBrisSpdRtg', 'BestBrisSpdRtg', 
    'AverageBrisSpdRtg', 'MedianBrisSpdRtg', 'BestLifeSpd', 'BestYearSpd', 'BestTrkSpd', 'WagerPost', 
    'Spd2F', 'Spd4F', 'SpdLate', 'Purse', 
    'LastFinishPos', 
    'LastBtnLns', 'LastDist', 'LastPurse', 'Finish2ndLastPos', 'Last2ndBtnLns', 'Last2ndDist', 'Last2ndPurse', 
    'FldSz', 'ML', 'FinishPos', 'BtnLns', 'AvgLast3', 'Exacta', 'TriFecta', 'NLANorm', 'WgtNorm', 
    'LastSpdRtgNorm', 'BestSpdRtgNorm', 'AverageSpdRtgNorm', 'MedianSpdRtgNorm', 'LastBrisSpdRtgNorm', 
    'BestBrisSpdRtgNorm', 'AverageBrisSpdRtgNorm', 'MedianBrisSpdRtgNorm', 'BestLifeSpdNorm', 'BestYearSpdNorm', 
    'BestTrkSpdNorm', 'WagerPostNorm', 'Spd2FNorm', 'Spd4FNorm', 'SpdLateNorm', 'MLNorm', 'FinishPosNorm', 
    'LowestSpdRtg', 'LastFldSz', 'SecondFldSz', 'ThirdFldSz', 'ThirdFinPos', 'FourthFldSz', 'FourthFinPos', 
    'ThirdPurse', 'FourthPurse', 'LastRaceDate', 'Last2ndRaceDate', 'Checked', 'Normalized', 'WinnersTime', 
    'ActualTime', 'MLRank', 'JockeyNorm', 'TrainerNorm', 'ActualFinish', 'ActualBtnLns', 
    'ActualOdds', 'ActualPurse'
]

text_fields = ['Surface']

# Exclude the ignored columns
text_fields = [col for col in text_fields if col not in ignore_columns]

for field in text_fields:
    df[field + '_sentiment'] = df[field].apply(calculate_sentiment)

numeric_cols = df.select_dtypes(include=['int', 'float']).columns
# Exclude the ignored columns
numeric_cols = [col for col in numeric_cols if col not in ignore_columns]

df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(df[numeric_cols].mean())

# Create a list of column names for numeric columns and sentiment scores
final_cols = list(numeric_cols) + [f + '_sentiment' for f in text_fields]

# Split data into X and y using only the final_cols
X = df[final_cols]
y = df['ActualFinish']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to float type
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

# Define model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Load new race data
new_race = pd.read_csv('c:\\races\\race1testdata.csv', encoding='utf-8-sig')
new_race['ActualFinish'].fillna(-1, inplace=True)
new_race['ActualTime'].fillna(-1, inplace=True)

# One-hot encoding
new_race = pd.get_dummies(new_race, columns=['Name'])

# Preprocessing
for field in text_fields:
    new_race[field + '_sentiment'] = new_race[field].apply(calculate_sentiment)

# Check if all the required columns exist in the new_race dataframe
missing_cols = [col for col in final_cols if col not in new_race.columns]
if missing_cols:
    print("Warning: The following required columns are missing from the new_race data: ", missing_cols)
    print("Adding missing columns with default value of 0.")
    for col in missing_cols:
        new_race[col] = 0

numeric_cols = new_race.select_dtypes(include=['int', 'float']).columns
# Exclude the ignored columns
numeric_cols = [col for col in numeric_cols if col not in ignore_columns]

new_race[numeric_cols] = new_race[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(new_race[numeric_cols].mean())

# Predict on new data
predictions = model.predict(new_race[final_cols].astype(np.float32))

# Add predicted finish positions to the new_race dataframe
new_race['PredictedFPos'] = predictions

# Sort the horses based on predicted finish positions
new_race = new_race.sort_values('PredictedFPos')

# Display the order of finish for all the horses
print("Order of finish:")
horse_name_cols = [col for col in new_race.columns if col.startswith('Name_')]
sorted_horse_name_cols = sorted(horse_name_cols, key=lambda x: new_race[x].iloc[0], reverse=True)
for i, col in enumerate(sorted_horse_name_cols):
    horse_name = col.split('_', 1)[1]
    print(f"Position {i+1}: {horse_name}")


    

