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
  - Park factors for 1B, 2B, 3B, HR, BB [done]
- Game state
  - Outs [done]
  - Innings [done]
  - Net Score [done]
  - 1B Occupied [done]
  - 2B Occupied [done]
  - 3B Occupied [done]
  - top/bottom of inning [done]
  - days since start of season [done]
  - temperature at game start time [not available yet]
- Batter fielding position
  - batter's 9 main positions + DH [TODO]


TODO:
- Check Team abbreviations for every year [done]
- Add logging 
- Time the transformers (preparation for below) [done]
- Scale the transformers by parallelizing jobs
  - concurrent futures [done]
  - Spark pipeline on databrick 
    - setup Spark locally 
    - Build Spark pipeline
- Offline model and feature selection in notebook 
- write script to scale to 20 years worth of data [done]
- translate the pipeline running code from notebook to prod [done]

