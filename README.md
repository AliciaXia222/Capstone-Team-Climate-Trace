# 🌎 Climate TRACE <br> Estimating Global Greenhouse Gas Emissions from Buildings
### Overview

**Name of Project**: Climate TRACE: Estimating global greenhouse gas emissions from buildings using machine learning and satellite imagery


**Contact**: Kyle Bradbury, Electrical and Computer Engineering at the Pratt School of Engineering; Nicholas School of the Environment; and Nicholas Institute for Energy Environment & Sustainability


**Summary**:​ Globally, buildings account for 30% of end-use energy consumption and 27% of energy sector greenhouse gas emissions, and yet the building sector is lacking in low-temporal latency, high-spatial-resolution data on energy consumption and resulting emissions. Existing methods tend to either have low resolution, high latency (often a year or more), or rely on data typically unavailable at scale (such as self-reported energy consumption). We will investigate machine learning based techniques that combine various features including those derived from satellite imagery to estimate global emissions estimates both for residential and commercial buildings at a 1km2 resolution. 
A more detailed vision for this work is described in this paper: https://www.climatechange.ai/papers/neurips2023/128/paper.pdf


**Goals**​: The team will create new feature inputs for a machine learning model based on characteristics such as temperature, humidity, economic data, population, and other factors related to building energy consumption. These features will be used to estimate energy use intensity, the key factor in determining a building's emissions intensity. These results will be compared to ground truth data and the performance evaluated. The team will also develop uncertainty estimates on their predictions to contextualize findings for the decision makers that will use the resulting data.

**Repository Directory Structure**

```python

├── README.md
├── data
│   ├── interim
│   │   └── HDD.csv
│   ├── processed
│   │   └── merged_df.csv
│   └── raw
│       ├── HDI_educationalIndex_incomeIndex.csv
│       ├── download.nc
│       ├── gdp_data.csv
│       └── population.csv
├── notebooks
│   ├── 01_DataPreprocessing.ipynb
│   ├── 02_Model.ipynb
│   └── 03_Plots.ipynb
├── reports
└── results
    ├── results_20241108_2123_all_domain.csv
    ├── results_20241108_2123_cross_domain.csv
    ├── results_20241108_2123_total.csv
    └── results_20241108_2123_within_domain.csv


```
