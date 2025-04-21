from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os
import re
from pathlib import Path



pd.set_option('display.max_columns', None)  # Show all columns

dataframes = []
folder_path = '/Users/jordangunti/Documents/PremierLeagueML/CSVs/'
folder = Path(folder_path)
def file_key(file):
    name = file.name
    if name == "E0.csv":
        return 0
    match = re.match(r"E0 \((\d+)\)\.csv", name)
    return int(match.group(1)) if match else float('inf')  # Skip anything else

files = sorted(folder.glob('E0*.csv'), key=file_key)

for file in files:
	if file.name.endswith('.csv') and file_key(file) <=8:
		df = pd.read_csv(file)
		dataframes.append(df)

df_all = pd.concat(dataframes,ignore_index=True)

df_all = df_all.dropna(axis=1, how='all')


result_map = {'H': 1,'D':0,'A': 2}
df_all['MatchResultsNumeric'] = df_all['FTR'].map(result_map)

home_stats = df_all.groupby('HomeTeam').agg(
    home_goals_scored=('FTHG', 'mean'),
    home_goals_conceded=('FTAG', 'mean')
).rename_axis('Team')

away_stats = df_all.groupby('AwayTeam').agg(
    away_goals_scored=('FTAG', 'mean'),  # FTAG = goals scored by away team
    away_goals_conceded=('FTHG', 'mean')  # FTHG = goals conceded by away team
).rename_axis('Team')


team_stats = pd.merge(home_stats, away_stats, on='Team', how='outer')
team_stats.index.name = 'Team'
team_stats.reset_index(inplace=True)
team_stats['Team'] = team_stats['Team'].astype(str)
# Merge team stats into df_all before training
df_all = df_all.merge(team_stats, left_on='HomeTeam', right_on='Team', how='left')
df_all = df_all.merge(team_stats, left_on='AwayTeam', right_on='Team', how='left', suffixes=('_home', '_away'))


df_2001 = pd.read_csv(folder_path + 'E0 (8).csv')
df_2001['HomeTeam'] = df_2001['HomeTeam'].astype(str)
df_2001 = df_2001.dropna(axis=1, how='all')
df_2001['MatchResultsNumeric'] = df_2001['FTR'].map(result_map)
df_2001 = df_2001.merge(team_stats, left_on='HomeTeam', right_on='Team', how='left')
df_2001 = df_2001.merge(team_stats, left_on='AwayTeam', right_on='Team', how='left', suffixes=('_home', '_away'))



X = df_all[[
    'home_goals_scored_home',
    'home_goals_conceded_home',
    'away_goals_scored_away',
    'away_goals_conceded_away'
]]
y = df_all['MatchResultsNumeric']  # Target variable


# Drop any rows where feature columns are NaN
X = X.dropna()
y = y.loc[X.index]  # Align y with X after dropping NaNs

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
