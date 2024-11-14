# 🌎 Climate TRACE <br> Estimating Global Greenhouse Gas Emissions from Buildings


## Table of Contents
1. [Abstract](#Abstract)
2. [Introduction](#Introduction)
3. [Methods](#Methods)
4. [Results](#Results)  

## Abstract

## Introduction

Global warming is one of the most critical challenges of our time, and addressing it requires accurately identifying the main sources of greenhouse gas (GHG) emissions. [Climate TRACE](https://climatetrace.org/), a global non-profit coalition, has made significant progress in independently tracking emissions with a high level of detail, covering approximately 83% of global emissions. However, the building sector, which represents a substantial portion of global energy consumption and GHG emissions, lacks timely, high-resolution, low-latency data on energy use and related emissions. Current methods are often outdated, with data available only after a year or more, or rely on self-reported information that is not available on a global scale. This data gap limits policymakers’ ability to focus their efforts effectively.

To address this challenge, we will develop a model to estimate GHG emissions based on building energy consumption. First, we will construct a model to predict energy use intensity (EUI), a key metric that measures energy consumption per unit area. Using EUI estimates along with global building floor area data provided by Climate TRACE, we can calculate GHG emissions based on formulas from previous research.

Our approach combines machine learning techniques with predictive variables derived from publicly available data, including meteorological factors such as temperature and humidity, as well as socioeconomic and demographic indicators. With these input features, our machine learning model will predict EUI at specific points within our initial dataset. Once validated, the model can be extrapolated to global locations, generating EUI and GHG emissions predictions at a 1 km² resolution. This approach enables more timely results, as many of our predictive variables—such as meteorological factors—are updated more frequently than self-reported data.

## Methods 
### Identifying Features



## Results  

### Feature Importance

![Image](https://raw.githubusercontent.com/AliciaXia222/Capstone-Team-Climate-Trace/refs/heads/main/figures/feature_importance.png)

## we
![Image](https://raw.githubusercontent.com/AliciaXia222/Capstone-Team-Climate-Trace/refs/heads/main/figures/experimental_design.png)

## Resources
1. [Markakis, P. J., Gowdy, T., Malof, J. M., Collins, L., Davitt, A., Volpato, G., & Bradbury, K. (2023). High-resolution global building emissions estimation using satellite imagery.](https://www.climatechange.ai/papers/neurips2023/128/paper.pdf)
2. 


## Contributors
[Jiechen Li](https://github.com/carrieli15)  
[Meixiang Du](https://github.com/dumeixiang)  
[Yulei Xia](https://github.com/AliciaXia222)  
[Barbara Flores](https://github.com/BarbaraPFloresRios)  



#### Project Mentor and Client: [Dr. Kyle Bradbury](https://energy.duke.edu/about/staff/kyle-bradbury)


