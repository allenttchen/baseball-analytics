# baseball-analytics

Goal: Predict matchup outcome for MLB games from scratch

Model: Neural Network with Pytorch

## Features
- Batter 365 days moving average (long-term)
  - Rates per PA for 1B, 2B, 3B, HR, BB, SO, DP, FO, HBP, SF, SH
  - Average wOBA 
  - Max exit velocity, Average exit velocity 
  - Number of plate appearances
- Pitcher 365 days moving average (long-term)
  - Rates per PA for 1B, 2B, 3B, HR, BB, SO, DP, FO, HBP, SF, SH
  - Average wOBA 
  - Max exit velocity, Average exit velocity 
  - Number of plate appearances
- Batter 30 days moving average (short-term)
- Pitcher 30 days moving average (short-term)
- Batter and Pitcher head-to-head all-time
- Ballpark
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
