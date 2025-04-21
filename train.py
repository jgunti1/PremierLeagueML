#from sklearn.preprocessing import LabelEncoder
import pandas as pd


df = pd.read_csv('/Users/jordangunti/Documents/PremierLeagueML/CSVs/E0.csv')

df = df.dropna(axis=1, how='all')

home_team_stats = df.groupby('HomeTeam').agg(
	avg_home_goals_scored=('FTHG','mean'),
	avg_home_goals_conceded=('FTAG','mean')
)
away_team_stats = df.groupby('AwayTeam').agg(
	avg_away_goals_scored=('FTHG','mean'),
	avg_away_goals_conceded=('FTAG','mean')
)
team_stats = home_team_stats.merge(away_team_stats,left_on='HomeTeam',right_on='AwayTeam')
print(team_stats)
