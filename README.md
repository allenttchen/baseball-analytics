# baseball-analytics

Goal: Predict matchup outcome for MLB games from scratch

Model: Neural Network with Pytorch

## Features
- Batter 365 days moving average (long-term)
  - Rates per PA for 1B, 2B, 3B, HR, BB, SO, DP, FO, HBP, SF, SH [done]
  - Average wOBA [done]
  - Max exit velocity, Average exit velocity [done]
  - Number of plate appearances [done]
  - Platoon wOBA, Number of platoon plate appearances [done]
- Pitcher 365 days moving average (long-term)
  - Rates per PA for 1B, 2B, 3B, HR, BB, SO, DP, FO, HBP, SF, SH [done]
  - Average wOBA [done]
  - Max exit velocity, Average exit velocity [done]
  - Number of plate appearances [done]
  - Platoon wOBA, Number of platoon plate appearances [done]
- Batter 30 days moving average (short-term)
  - Rates per PA for 1B, 2B, HR, BB, SO [done]
  - Number of plate appearances [done]
  - Average wOBA [done]
- Pitcher 30 days moving average (short-term)
  - Rates per PA for 1B, 2B, HR, BB, SO [done]
  - Number of plate appearances [done]
  - Average wOBA [done]
- Batter and Pitcher head-to-head all-time
  - Rates per PA for 1B, 2B, HR, BB, SO [done]
  - Average wOBA [done]
  - Number of appearances [done]
- Ballpark
  - Park factors for 1B, 2B, 3B, HR, BB
- Game state
  - Outs
  - Innings
  - Net Score
  - 1B Occupied
  - 2B Occupied
  - 3B Occupied
  - top/bottom of inning
  - days since start of season
  - temperature at game start time
- Batter fielding position
  - batter's 9 main positions + DH
