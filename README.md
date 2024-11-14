# 🌎 Climate TRACE <br> Estimating Global Greenhouse Gas Emissions from Buildings


## Table of Contents
1. [Abstract](#Abstract)
2. [Introduction](#Introduction)
3. [Methods](#Methods)  
   3.1 [Identifying Features](#IdentifyingFeatures)  
   3.2 [Experimental Design](#ExperimentalDesign) 
4. [Results](#Results)  
   3.1 [Identifying Features](#IdentifyingFeatures)  
   3.2 [Model Results](#ModelResults)  


7. [Resources](#Resources)  
8. [Contributors](#Contributors)


## Abstract <a name="Abstract"></a>

## Introduction <a name="Introduction"></a>

Globally, buildings account for 30% of end-use energy consumption and 27% of energy sector greenhouse gas emissions, and yet the building sector is lacking in low-temporal latency, high-spatial-resolution data on energy consumption and resulting emissions. Existing methods tend to either have low resolution, high latency (often a year or more), or rely on data typically unavailable at scale (such as self-reported energy consumption). We will investigate machine learning based techniques that combine various features including those derived from satellite imagery to estimate global emissions estimates both for residential and commercial buildings at a 1km2 resolution. 
A more detailed vision for this work is described in this paper: https://www.climatechange.ai/papers/neurips2023/128/paper.pdf


The team will create new feature inputs for a machine learning model based on characteristics such as temperature, humidity, economic data, population, and other factors related to building energy consumption. These features will be used to estimate energy use intensity, the key factor in determining a building's emissions intensity. These results will be compared to ground truth data and the performance evaluated. The team will also develop uncertainty estimates on their predictions to contextualize findings for the decision makers that will use the resulting data.




![Geographic Distribution of Data Points by Region](https://raw.githubusercontent.com/AliciaXia222/Capstone-Team-Climate-Trace/refs/heads/main/figures/region_map.png)

## Methods <a name="Methods"></a>
### Identifying Features <a name="IdentifyingFeatures"></a>

### Experimental Design <a name="ExperimentalDesign"></a>

![Image](https://raw.githubusercontent.com/AliciaXia222/Capstone-Team-Climate-Trace/refs/heads/main/figures/experimental_design.png)


## Results  <a name="Results"></a>

### Feature Importance <a name="Feature Importance"></a>

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


