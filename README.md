# 🌎 Estimating Global Greenhouse Gas Emissions from Buildings

## Table of Contents
1. [Abstract](#Abstract)
2. [Introduction](#Introduction)
3. [Methods](#Methods)  
   3.1 [Identifying Features](#IdentifyingFeatures)  
   3.2 [Experimental Design](#ExperimentalDesign) 
4. [Results](#Results)  
   4.1 [Feature Importance](#FeatureImportance)  
   4.2 [Model Results](#ModelResults)
5. [Resources](#Resources)  
6. [Contributors](#Contributors)


## Abstract <a name="Abstract"></a>

This project develops a machine learning model to estimate greenhouse gas (GHG) emissions from building energy consumption. By predicting energy use intensity (EUI) using variables such as heating degree days (HDD), humidity, Human Development Index (HDI), educational index, income index, and GDP per capita, the model will generate estimates of energy consumption for both residential and non-residential buildings. These EUI estimates, along with global building floor area data provided by our client, will be used to calculate GHG emissions, offering a timely, high-resolution approach to estimating emissions at a global scale.


## Introduction <a name="Introduction"></a>

Global warming is one of the most critical challenges of our time, and addressing it requires accurately identifying the main sources of greenhouse gas (GHG) emissions. Climate TRACE, a global non-profit coalition, has made significant progress in independently tracking emissions with a high level of detail, covering approximately 83% of global emissions. However, the building sector, which represents a substantial portion of global energy consumption and GHG emissions, lacks timely, high-resolution, low-latency data on energy use and related emissions. Current methods are often outdated, with data available only after a year or more, or rely on self-reported information that is not available on a global scale. This data gap limits policymakers’ ability to focus their efforts effectively.

Our project focuses on emissions from the building sector. Buildings contribute between 6% and 9% of global emissions when considering only direct emissions, which primarily result from onsite fossil fuel combustion used for space heating, water heating, and cooking. Emissions from lighting, consumer electronics, and most air conditioning are excluded, as these are typically electric and accounted for separately within Climate TRACE.

This project is focused on developing a machine learning model to estimate GHG emissions based on building energy consumption. The model will predict energy use intensity (EUI) using predictive variables such as temperature, humidity, and socioeconomic data, along with global building floor area data from Climate TRACE. These EUI estimates, along with building area data, will be used to calculate direct GHG emissions, providing building emissions data in 1-kilometer-by-1-kilometer grid cells.

## Methods <a name="Methods"></a>

### Building Emissions Estimation <a name="BuildingEmissionsEstimation"></a>

To estimate greenhouse gas (GHG) emissions from buildings, we will use Energy Use Intensity (EUI) as a central metric. EUI measures the energy consumption per square meter of building space, making it a valuable indicator for emissions estimation. By combining EUI values with total building floor area and an emissions factor, we can calculate the GHG emissions associated with buildings.

The estimation formula is:
![Formula](/figures/formula.png)

### Identifying Features <a name="IdentifyingFeatures"></a>

Since we aim to predict energy use intensity (EUI) for buildings, the focus is primarily on direct emissions. These emissions largely result from onsite fossil fuel combustion used for space heating, water heating, and cooking. To represent these factors through data, we identified the following datasets from open resources that align with our requirements:

1. **EUI [Google Drive](https://drive.google.com/uc?id=12qGq_DLefI1RihIF_RKQUyJtm480-xRC)**: This serves as our ground truth data for energy use intensity, provided by the client. It contains 482 rows and two key columns:  
   - *Residential EUI*: Calculated based on the area of residential buildings.  
   - *Non-Residential EUI*: Calculated based on the area of non-residential buildings.  

2. **Temperature [Copernicus](https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics?tab=overview)**: This dataset provides daily temperature statistics, offering insights into climate-related factors that influence energy use.

3. **Population [World Bank Group](https://data.worldbank.org/indicator/SP.POP.TOTL)**: This dataset includes population data for various countries and regions from 1960 to 2023. For our analysis, we extracted the population figures for 2023 to align with our project goals.

4. **GDP [Global Data Lab](https://globaldatalab.org/shdi/metadata/shdi/)**: This dataset contains data on human development, health, education, and income across 160+ countries from 1990 to 2022. We used the GDP values for 2022 as a key feature for our model.

5. **Human Development Index (HDI) [Global Data Lab](https://globaldatalab.org/shdi/metadata/shdi/)**: HDI measures a country's achievements in three key areas:  
   - *Health*: A long and healthy life.  
   - *Knowledge*: Access to education.  
   - *Standard of Living*: A decent standard of living.  
   We extracted data for the year 2022 to maintain consistency with other datasets.

6. **Urbanization Rate [World Bank](https://data.worldbank.org/indicator/SP.URB.TOTL.IN.ZS?end=2023&start=2023&view=map&year=2022)**: Urbanization rate reflects the average annual growth of urban populations. For consistency, we used data from 2022.

7. **Educational Index [Global Data Lab](https://globaldatalab.org/shdi/metadata/edindex/)**: This index comprises two indicators:  
   - *Mean Years of Schooling (MYS)*: The average years of schooling for adults aged 25 and above.  
   - *Expected Years of Schooling (EYS)*: The anticipated years of education for the current population.  

8. **Paris Agreement [United Nations Climate Change](https://unfccc.int/process-and-meetings/the-paris-agreement)**: The Paris Agreement is an international treaty adopted by 196 parties in 2015. We used this information to create a binary variable (`Paris_Agreement`) to indicate whether a country is a signatory.

9. **Humidity [Copernicus](https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics?tab=overview)**: Humidity was measured using the dew point temperature at 2 meters above ground. The dew point is a reliable measure of how "dry" or "humid" conditions feel, making it preferable over relative humidity for capturing human comfort levels.

10. **Latitude [GeoNames](https://download.geonames.org/export/dump/)**: This dataset provides global latitude data in decimal degrees (WGS84 coordinate reference system), adding geographical context to our analysis.

11. **Longitude [GeoNames](https://download.geonames.org/export/dump/)**: This dataset provides global longitude data in decimal degrees (WGS84 coordinate reference system), complementing the latitude data for geographical analysis.

![Diagram](/figures/diagram.png)

### Experimental Design <a name="ExperimentalDesign"></a>

![Geographic Distribution of Data Points by Region](/figures/region_map.png)


![Image](/figures/experimental_design.png)


## Results  <a name="Results"></a>

### Feature Importance <a name="FeatureImportance"></a>

To identify the most influential variables in building energy consumption and their greenhouse gas emissions, we used a linear regression model, which allows us to directly assess the relevance of each variable in predicting energy use intensity (EUI). Among all the features, Heating Degree Days (HDD), defined as a measure of heating demand based on temperature, proved to be the most significant factor, highlighting the importance of temperature in energy consumption. This suggests that, in future iterations of the model, it would be useful to explore temperature-related variables, such as average temperature, along with humidity, to improve the estimation of emissions in the building sector.

![Feature Importance](/figures/feature_importance.png)

### Model Results <a name="ModelResults"></a>

## Resources  <a name="Resources"></a>
1. [Greet Janssens-Maenhout, Monica Crippa, Diego Guizzardi, Marilena Muntean, Edwin Schaaf, Frank Dentener, Peter Bergamaschi, Valerio Pagliari, Jos G. J. Olivier, Jeroen A. H. W. Peters, John A. van Aardenne, Suvi Monni, Ulrike Doering, A. M. Roxana Petrescu, Efisio Solazzo, and Gabriel D. Oreggioni. (July 2019). EDGARv4.3.2 Global Atlas of the three major greenhouse gas emissions for the period 1970–2012.](https://essd.copernicus.org/articles/11/959/2019/)
2. [Markakis, P. J., Gowdy, T., Malof, J. M., Collins, L., Davitt, A., Volpato, G., & Bradbury, K. (2023). High-resolution global building emissions estimation using satellite imagery.](https://www.climatechange.ai/papers/neurips2023/128/paper.pdf)
3. [Marshall Burke*, Anne Driscoll, David B. Lobell, Stefano Ermon. (2021). Using satellite imagery to understand and promote sustainable development.](https://www.science.org/doi/full/10.1126/science.abe8628)
4. [Xinyu Dou, Yilong Wang, Philippe Ciais, Frédéric Chevallier, Steven J. Davis, Monica Crippa, Greet Janssens-Maenhout, Diego Guizzardi, Efisio Solazzo, Feifan Yan, Da Huo, Bo Zheng, Biqing Zhu, Duo Cui, Piyu Ke, Taochun Sun, Hengqi Wang, Qiang Zhang, Pierre Gentine, Zhu Deng, and Zhu Liu. (2022). Near-realtime global gridded daily CO2 emissions.](https://www.sciencedirect.com/science/article/pii/S2666675821001077)


## Contributors  <a name="Contributors"></a>
[Jiechen Li](https://github.com/carrieli15)  
[Meixiang Du](https://github.com/dumeixiang)  
[Yulei Xia](https://github.com/AliciaXia222)  
[Barbara Flores](https://github.com/BarbaraPFloresRios)  



#### Project Mentor and Client: [Dr. Kyle Bradbury](https://energy.duke.edu/about/staff/kyle-bradbury)


