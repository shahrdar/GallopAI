import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def calculate_sentiment(text):
    if text == 'D':
        return 1
    elif text == 'T':
        return 0
    else:
        return 0

df = pd.read_csv('c:\\races\\penraces2022.csv', encoding='utf-8-sig', low_memory=False)

# Convert column 21 to a uniform type (string as an example)
df.iloc[:, 21] = df.iloc[:, 21].astype(str)

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

text_fields = [col for col in text_fields if col not in ignore_columns]

# Calculate sentiment and replace the original columns with these sentiment values
for field in text_fields:
    sentiment_field = field + '_sentiment'
    if sentiment_field not in df.columns:
        df[sentiment_field] = df[field].apply(calculate_sentiment)

    # Replace original column with its sentiment version
    df[field] = df[sentiment_field]

numeric_cols = df.select_dtypes(include=['int', 'float']).columns
numeric_cols = [col for col in numeric_cols if col not in ignore_columns]
numeric_cols = [col for col in numeric_cols if col not in text_fields]

df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(df[numeric_cols].mean())

final_cols = list(numeric_cols)

X = df[final_cols]
y = df['ActualFinish']

# Check for any remaining NaNs or infinities
assert not X.isnull().values.any(), "There are still NaNs in the input data."
assert np.isfinite(X.to_numpy()).all(), "There are still infinities in the input data."

# Normalize target
scaler = StandardScaler()
y = scaler.fit_transform(y.values.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.float32)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

new_race = pd.read_csv('c:\\races\\race1testdata.csv', encoding='utf-8-sig', low_memory=False)
new_race['ActualFinish'].fillna(-1, inplace=True)
new_race['ActualTime'].fillna(-1, inplace=True)

new_race = pd.get_dummies(new_race, columns=['Name'])

for field in text_fields:
    sentiment_field = field + '_sentiment'
    if sentiment_field not in new_race.columns:
        new_race[sentiment_field] = new_race[field].apply(calculate_sentiment)


missing_cols = [col for col in final_cols if col not in new_race.columns]
if missing_cols:
    print("Warning: The following required columns are missing from the new_race data: ", missing_cols)
    print("Adding missing columns with default value of 0.")
    for col in missing_cols:
        new_race[col] = 0

numeric_cols = new_race.select_dtypes(include=['int', 'float']).columns
numeric_cols = [col for col in numeric_cols if col not in ignore_columns]
new_race[numeric_cols] = new_race[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(new_race[numeric_cols].mean())

predictions = model.predict(new_race[final_cols].astype(np.float32))

new_race['PredictedFPos'] = predictions
new_race = new_race.sort_values('PredictedFPos')

print("Order of finish:")
horse_name_cols = [col for col in new_race.columns if col.startswith('Name_')]
sorted_horse_name_cols = sorted(horse_name_cols, key=lambda x: new_race[x].iloc[0], reverse=True)
for i, col in enumerate(sorted_horse_name_cols):
    horse_name = col.split('_', 1)[1]
    print(f"Position {i+1}: {horse_name}")
