# 🌎 Estimating Global Greenhouse Gas Emissions from Buildings

## Table of Contents
1. [Abstract](#Abstract)
2. [Introduction](#Introduction)
3. [Methods](#Methods)  
   3.1 [Identifying Features](#IdentifyingFeatures)  
   3.2 [Experimental Design](#ExperimentalDesign) 
4. [Results](#Results)  
   4.1 [Identifying Features](#IdentifyingFeatures)  
   4.2 [Model Results](#ModelResults)
5. [Resources](#Resources)  
6. [Contributors](#Contributors)


## Abstract <a name="Abstract"></a>
This project develops a machine learning model to estimate greenhouse gas (GHG) emissions from building energy consumption. By predicting energy use intensity (EUI) using variables such as heating degree days (HDD), humidity, Human Development Index (HDI), educational index, income index, and GDP per capita, the model will generate estimates of energy consumption for both residential and non-residential buildings. These EUI estimates, along with global building floor area data provided by our client, will be used to calculate GHG emissions, offering a timely, high-resolution approach to estimating emissions at a global scale.


## Introduction <a name="Introduction"></a>

Global warming is one of the most critical challenges of our time, and addressing it requires accurately identifying the main sources of greenhouse gas (GHG) emissions. Climate TRACE, a global non-profit coalition, has made significant progress in independently tracking emissions with a high level of detail, covering approximately 83% of global emissions. However, the building sector, which represents a substantial portion of global energy consumption and GHG emissions, lacks timely, high-resolution, low-latency data on energy use and related emissions. Current methods are often outdated, with data available only after a year or more, or rely on self-reported information that is not available on a global scale. This data gap limits policymakers’ ability to focus their efforts effectively.

This project develops a machine learning model to estimate GHG emissions based on building energy consumption. The model will predict energy use intensity (EUI) using predictive variables such as temperature, humidity, and socioeconomic data, along with global building floor area data from Climate TRACE. These EUI estimates will be used to calculate GHG emissions.


## Methods <a name="Methods"></a>

### Building Emissions Estimation <a name="BuildingEmissionsEstimation"></a>

To estimate greenhouse gas (GHG) emissions from buildings, we will use Energy Use Intensity (EUI) as a central metric. EUI measures the energy consumption per square meter of building space, making it a valuable indicator for emissions estimation. By combining EUI values with total building floor area and an emissions factor, we can calculate the GHG emissions associated with buildings.

The estimation formula is:
![Formula](/figures/formula.png)

### Identifying Features <a name="IdentifyingFeatures"></a>

![Diagram](/figures/diagram.png)


### Experimental Design <a name="ExperimentalDesign"></a>

![Geographic Distribution of Data Points by Region](/figures/region_map.png)


![Image](/figures/experimental_design.png)


## Results  <a name="Results"></a>

### Feature Importance <a name="Feature Importance"></a>

To identify the most influential variables in building energy consumption and their greenhouse gas emissions, we used a linear regression model, which allows us to directly assess the relevance of each variable in predicting energy use intensity (EUI). Among all the features, Heating Degree Days (HDD) proved to be the most significant, highlighting the importance of temperature in energy consumption. This suggests that, in future iterations of the model, it would be useful to explore temperature-related variables, such as average temperature, along with humidity, to improve the estimation of emissions in the building sector.

![Feature Importance](/figures/feature_importance.png)

### Model Results <a name="ModelResults"></a>

## Resources  <a name="Resources"></a>
1. [Markakis, P. J., Gowdy, T., Malof, J. M., Collins, L., Davitt, A., Volpato, G., & Bradbury, K. (2023). High-resolution global building emissions estimation using satellite imagery.](https://www.climatechange.ai/papers/neurips2023/128/paper.pdf)
2. 


## Contributors  <a name="Contributors"></a>
[Jiechen Li](https://github.com/carrieli15)  
[Meixiang Du](https://github.com/dumeixiang)  
[Yulei Xia](https://github.com/AliciaXia222)  
[Barbara Flores](https://github.com/BarbaraPFloresRios)  



#### Project Mentor and Client: [Dr. Kyle Bradbury](https://energy.duke.edu/about/staff/kyle-bradbury)


