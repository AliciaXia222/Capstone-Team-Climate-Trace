# 🌎 Estimating Global Greenhouse Gas Emissions from Buildings

## Table of Contents

1. [Abstract](#Abstract)
2. [Problem Statement and Project Motivation](#ProblemStatement)
3. [Introduction](#Introduction)
4. [Background](#Background)  
5. [Data Description](#DataDescription)  
   5.1 [Building Emissions Estimation](#BuildingEmissionsEstimation)  
   5.2 [Feature Description](#FeatureDescription)  
6. [Methods](#Methods)  
   6.1 [Feature Engineering](#FeatureEngineering)  
   6.2 [Nearest Reference Mapping](#NearestReferenceMapping)  
   6.3 [Supervised Machine Learning](#SupervisedMachineLearning)  
7. [Experiments](#Experiments)  
   7.1 [Experimental Design](#ExperimentalDesign)  
8. [Conclusion](#Conclusion)  
   8.1 [Feature Importance](#FeatureImportance)  
   8.2 [Model Results](#ModelResults)  
9. [Resources](#Resources)
10. [Repository Structure and Usage](#RepositoryStructureAndUsage)
11. [Contributors](#Contributors) 


## 1.Abstract <a name="Abstract"></a>

This project develops a machine learning model to estimate greenhouse gas (GHG) emissions from building energy consumption. By predicting energy use intensity (EUI) using variables such as heating degree days (HDD), humidity, Human Development Index (HDI), educational index, income index, and GDP per capita, the model will generate estimates of energy consumption for both residential and non-residential buildings. These EUI estimates, along with global building floor area data provided by our client, will be used to calculate GHG emissions, offering a timely, high-resolution approach to estimating emissions at a global scale.

## 2.Problem Statement and Project Motivation <a name="ProblemStatement"></a>

The building sector lacks timely, high-resolution, and low-latency data on energy consumption and greenhouse gas (GHG) emissions, limiting efforts to address its significant contribution to global emissions. Current methods are often outdated, with data available only after a year or more, or rely on self-reported information that is not available on a global scale. This data gap severely restricts policymakers’ ability to focus their efforts effectively.

To bridge this gap, our project seeks to develop a methodology to estimate global onsite building emissions with high spatial resolution, specifically using a 1-kilometer-by-1-kilometer grid. This effort focuses on developing a machine learning-based approach to predict energy consumption and GHG emissions in near real-time, sharing open-source methodologies to ensure replicability and broader adoption, and validating models to assess uncertainty and reliability for global application.


## 3.Introduction <a name="Introduction"></a>

Global warming is one of the most critical challenges of our time, and addressing it requires accurately identifying the main sources of greenhouse gas (GHG) emissions. Climate TRACE, a global non-profit coalition, has made significant progress in independently tracking emissions with a high level of detail, covering approximately 83% of global emissions. However, the building sector, which represents a substantial portion of global energy consumption and GHG emissions, lacks timely, high-resolution, low-latency data on energy use and related emissions. Current methods are often outdated, with data available only after a year or more, or rely on self-reported information that is not available on a global scale. This data gap limits policymakers’ ability to focus their efforts effectively.

Our project focuses on emissions from the building sector. Buildings contribute between 6% and 9% of global emissions when considering only direct emissions, which primarily result from onsite fossil fuel combustion used for space heating, water heating, and cooking. Emissions from lighting, consumer electronics, and most air conditioning are excluded, as these are typically electric and accounted for separately within Climate TRACE.

This project is focused on developing a machine learning model to estimate GHG emissions based on building energy consumption. The model will predict energy use intensity (EUI) using predictive variables such as temperature, humidity, and socioeconomic data, along with global building floor area data from Climate TRACE. These EUI estimates, along with building area data, will be used to calculate direct GHG emissions, providing building emissions data in 1-kilometer-by-1-kilometer grid cells.

## 4.Background <a name="Background"></a>

Existing estimates of anthropogenic CO2 emissions are provided by several sources, including the Open-source Data Inventory for Anthropogenic CO2 [5], the Community Emissions Data System [6], the Emissions Database for Global Atmospheric Research (EDGAR) [7], the Global Carbon Grid [8], and the Global Gridded Daily CO2 Emissions Dataset (GRACED) [9]. While GRACED data is updated near-monthly, most of the other key datasets have a production latency of a year or more. Furthermore, the highest resolution available across these datasets is 0.1 decimal degrees, which corresponds to approximately 11 km near the equator. Additionally, only a few of these models provide a breakdown of emissions into residential and commercial subsectors, or offer separate emissions estimates for individual greenhouse gases.

## 5.Data Description <a name="DataDescription"></a>

### 5.1 Building Emissions Estimation <a name="BuildingEmissionsEstimation"></a>

To estimate greenhouse gas (GHG) emissions from buildings, we will use Energy Use Intensity (EUI) as a central metric. EUI measures the energy consumption per square meter of building space, making it a valuable indicator for emissions estimation. By combining EUI values with total building floor area and an emissions factor, we can calculate the GHG emissions associated with buildings.

The estimation formula is:
![Formula](/figures/formula.png)

### 5.2 Feature Description <a name="FeatureDescription"></a>

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

## 6.Methods <a name="Methods"></a>

### 6.1 Feature Engineering <a name="FeatureEngineering"></a>
Feature engineering is essential to transform raw data into meaningful representations that enhance model performance and predictive accuracy. In this study, we applied the following techniques:  

1. **Heading Degree Days Calculation:**  
   Calculated using temperature data to derive features measure the demand for heating energy based on the difference between outdoor temperature and a baseline "comfort" temperature, typically 65°F (18°C).  

2. **Comfort Index Calculation:**  
   Derived using temperature and humidity data to quantify and evaluate human thermal comfort, which is influenced by environmental factors like temperature and humidity.

3. **Cross-Feature Interaction:**  
   Combined multiple features to create new interaction terms that capture relationships between variables.  

4. **Clustering Features:**  
   Apply clustering algorithms (e.g., k-means) to group data points and used cluster labels as additional features for modeling.

### 6.2 Nearest Reference Mapping <a name="NearestReferenceMapping"></a>

Nearest Reference Mapping involves assign each data point to its closest reference location based on a defined distance metric, enriching the dataset with relevant features from these reference points. 

In this project, we aim to assigning **EUI values** to each data point based on its nearest starting point with known ground truth. By using the EUI values as features and incorporating spatial context into our model, we aim to improve the model’s starting point and enhance prediction accuracy for global projections. 

### 6.3 Supervised Machine Learning <a name="SupervisedMachineLearning"></a>  


In this project, we will employ a range of supervised machine learning models to predict and analyze the target variable. The following models will be utilized:

1. **Linear Regression:**  
   We will use Linear Regression to model the relationship between the input features and the target variable. This model is suitable for capturing linear relationships and will serve as a baseline for comparison with more complex models.

2. **K-Nearest Neighbors (KNN):**  
   KNN is a non-parametric model that classifies a data point based on the majority class or average value of its nearest neighbors. It is particularly useful for capturing local patterns in the data and will provide a comparison to Linear Regression in terms of flexibility.

3. **Ensemble Models:**
   - **XGBoost:**  
     XGBoost is an optimized gradient boosting algorithm that performs well in a variety of prediction tasks. It builds an ensemble of decision trees sequentially, improving the model’s performance by reducing bias and variance.
   
   - **CatBoost:**  
     CatBoost is another gradient boosting algorithm known for its handling of categorical features without the need for explicit preprocessing. It is expected to provide competitive results, particularly in datasets with mixed types of variables.

The combination of linear models, distance-based methods like KNN, and powerful ensemble models like XGBoost and CatBoost will allow us to capture a range of patterns in the data, from simple linear trends to more complex interactions and non-linear relationships.

### Experimental Design <a name="ExperimentalDesign"></a>
Given the challenge of regional variations in global data, we will validate our predictions at the regional level across 5 regions using 3 strategies to identify biases and improve model robustness.

![Geographic Distribution of Data Points by Region](/figures/region_map.png)

![Image](/figures/experimental_design.png)

We aim to assess our model's generalization by comparing its performance within the same region (Within-Domain) and its ability to extrapolate to other regions (Cross-Domain). The goal is to reduce the gap between these strategies to improve accuracy and understand extrapolation errors. Additionally, we want to understand if there are regions that perform better than others in specific outcomes, which can help us tailor our model to regional differences.


## 8. Results  <a name="Results"></a>

### Feature Importance <a name="FeatureImportance"></a>

To identify the most influential variables in building energy consumption and their greenhouse gas emissions, we used a linear regression model, which allows us to directly assess the relevance of each variable in predicting energy use intensity (EUI). Among all the features, Heating Degree Days (HDD), defined as a measure of heating demand based on temperature, proved to be the most significant factor, highlighting the importance of temperature in energy consumption. This suggests that, in future iterations of the model, it would be useful to explore temperature-related variables, such as average temperature, along with humidity, to improve the estimation of emissions in the building sector.

![Feature Importance](/figures/feature_importance.png)

### Model Results <a name="ModelResults"></a>



## 9. Resources  <a name="Resources"></a>
1. [Greet Janssens-Maenhout, Monica Crippa, Diego Guizzardi, Marilena Muntean, Edwin Schaaf, Frank Dentener, Peter Bergamaschi, Valerio Pagliari, Jos G. J. Olivier, Jeroen A. H. W. Peters, John A. van Aardenne, Suvi Monni, Ulrike Doering, A. M. Roxana Petrescu, Efisio Solazzo, and Gabriel D. Oreggioni. (July 2019). EDGARv4.3.2 Global Atlas of the three major greenhouse gas emissions for the period 1970–2012.](https://essd.copernicus.org/articles/11/959/2019/)
2. [Markakis, P. J., Gowdy, T., Malof, J. M., Collins, L., Davitt, A., Volpato, G., & Bradbury, K. (2023). High-resolution global building emissions estimation using satellite imagery.](https://www.climatechange.ai/papers/neurips2023/128/paper.pdf)
3. [Marshall Burke*, Anne Driscoll, David B. Lobell, Stefano Ermon. (2021). Using satellite imagery to understand and promote sustainable development.](https://www.science.org/doi/full/10.1126/science.abe8628)
4. [Xinyu Dou, Yilong Wang, Philippe Ciais, Frédéric Chevallier, Steven J. Davis, Monica Crippa, Greet Janssens-Maenhout, Diego Guizzardi, Efisio Solazzo, Feifan Yan, Da Huo, Bo Zheng, Biqing Zhu, Duo Cui, Piyu Ke, Taochun Sun, Hengqi Wang, Qiang Zhang, Pierre Gentine, Zhu Deng, and Zhu Liu. (2022). Near-realtime global gridded daily CO2 emissions.](https://www.sciencedirect.com/science/article/pii/S2666675821001077)
5. [Oda, T., Maksyutov, S., & Andres, R. J. (2018). The Open-source Data Inventory for Anthropogenic CO2, version 2016 (ODIAC2016): A global monthly fossil fuel CO2 gridded emissions data product for tracer transport simulations and surface flux inversions. Earth System Science Data, 10(1), 87–107.](https://doi.org/10.5194/essd-10-87-2018)
6. [McDuffie, E. E., Smith, S. J., O’Rourke, P., Tibrewal, K., Venkataraman, C., Marais, E. A., Zheng, B., Crippa, M., Brauer, M., & Martin, R. V. (2020). A global anthropogenic emission inventory of atmospheric pollutants from sector- and fuel-specific sources (1970–2017): An application of the Community Emissions Data System (CEDS). Earth System Science Data, 12(4), 3413–3442.](https://doi.org/10.5194/essd-12-3413-2020)
7. [Janssens-Maenhout, G., Crippa, M., Guizzardi, D., Muntean, M., Schaaf, E., Dentener, F., Bergamaschi, P., Pagliari, V., Olivier, J. G. J., Peters, J. A. H. W., van Aardenne, J. A., Monni, S., Doering, U., Petrescu, A. M. R., Solazzo, E., & Oreggioni, G. D. (2019). EDGAR v4.3.2 Global Atlas of the three major greenhouse gas emissions for the period 1970–2012. Earth System Science Data, 11(3), 959–1002.](https://doi.org/10.5194/essd-11-959-2019)
8. [Tong, D., Zhang, Q., Davis, S. J., Liu, F., Zheng, B., Geng, G., Xue, T., Li, M., Hong, C., Lu, Z., Streets, D. G., Guan, D., & He, K. (2018). Targeted emission reductions from global super-polluting power plant units. Nature Sustainability, 1(1), 59–68.](https://doi.org/10.1038/s41893-017-0003-y)
9. [Dou, X., Wang, Y., Ciais, P., Chevallier, F., Davis, S. J., Crippa, M., Janssens-Maenhout, G., Guizzardi, D., Solazzo, E., Yan, F., Huo, D., Zheng, B., Zhu, B., Cui, D., Ke, P., Sun, T., Wang, H., Zhang, Q., Gentine, P., Deng, Z., & Liu, Z. (2022). Near-realtime global gridded daily CO2 emissions. The Innovation, 3(1), 100182.](https://doi.org/10.1016/j.xinn.2021.100182)


## 10. Repository Structure and Usage <a name="RepositoryStructureAndUsage "></a>


## 11. Contributors  <a name="Contributors"></a>
[Jiechen Li](https://github.com/carrieli15)  
[Meixiang Du](https://github.com/dumeixiang)  
[Yulei Xia](https://github.com/AliciaXia222)  
[Barbara Flores](https://github.com/BarbaraPFloresRios)  



#### Project Mentor and Client: [Dr. Kyle Bradbury](https://energy.duke.edu/about/staff/kyle-bradbury)


