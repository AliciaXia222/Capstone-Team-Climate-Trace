# 🌎 Estimating Global Greenhouse Gas Emissions from Buildings

## Table of Contents

1. [Abstract](#Abstract)
2. [Introduction](#Introduction)  
   2.1 [Project Motivation](#ProjectMotivation)  
   2.2 [Problem Statement](#ProblemStatement)  
   2.3 [Goals](#Goals)  
3. [Background](#Background)  
4. [Data Description](#DataDescription)  
   4.1 [Building Emissions Estimation](#BuildingEmissionsEstimation)  
   4.2 [Feature Description](#FeatureDescription)  
5. [Methods](#Methods)  
   5.1 [Feature Engineering](#FeatureEngineering)  
   5.2 [Nearest Reference Mapping](#NearestReferenceMapping)  
   5.3 [Supervised Machine Learning](#SupervisedMachineLearning)  
6. [Experiments](#Experiments)  
   6.1 [Experimental Design](#ExperimentalDesign)  
7. [Conclusion](#Conclusion)  
   7.1 [Feature Importance](#FeatureImportance)  
   7.2 [Model Results](#ModelResults)  
8. [Resources](#Resources)
9. [Repository Structure and Usage](#RepositoryStructureAndUsage)
10. [Contributors](#Contributors) 


## 1. Abstract <a name="Abstract"></a>

This project develops a machine learning model to estimate direct greenhouse gas (GHG) emissions from residential and non-residential building energy consumption. The model predicts energy use intensity (EUI) by incorporating climatic, geographical, and socioeconomic variables for both residential and non-residential buildings. These EUI estimates, along with global building floor area, will be used in the next stage of this project to calculate direct GHG emissions from buildings, offering a timely, high-resolution method for global emissions estimation. This current work outlines preliminary EUI estimation techniques, while future iterations will refine the model by incorporating additional features to enhance performance, ultimately addressing the challenge of estimating global direct GHG emissions from buildings.

## 2. Introduction <a name="Introduction"></a>

### 2.1  Project Motivation  <a name="ProjectMotivation"></a>

Global warming is one of the most critical challenges of our time, and to address it effectively, we need more detailed information on where and when greenhouse gas emissions occur. This data is crucial for setting actionable emissions reduction goals and enabling policymakers to make informed decisions. Given this situation, Climate TRACE, a non-profit coalition of organizations, is building a timely, open, and accessible inventory of global emissions sources, currently covering around 83% of global emissions.

Building direct emissions are responsible for between 6% and 9% of global GHG emissions, primarily due to onsite fossil fuel combustion for heating, water heating, and cooking. Indirect emissions from lighting, consumer electronics, and air conditioning are excluded, as they are typically electric and accounted for separately in the Climate TRACE database.

Despite their significant contribution to global emissions, the building sector still lacks the timely, high-resolution, and low-latency data needed to assess GHG emissions accurately. Current methodologies rely on outdated data, often delayed by over a year, or on self-reported data that is scarce or unavailable globally.

### 2.2  Problem Statement  <a name="ProjectStatement"></a>

Specifically, we can define our problem statement as follows:

***The building sector lacks timely, high-resolution data on direct greenhouse gas (GHG) emissions, limiting the ability to accurately track and reduce emissions from building energy use.***

### 2.3  Goals  <a name="Goals"></a>

The goal of this project is to develop a machine learning model to estimate greenhouse gas (GHG) emissions based on building energy consumption. The model will predict energy use intensity (EUI) using climatic, geographical, and socioeconomic variables. These EUI estimates, along with building area data, will be used to calculate direct GHG building emissions.

In the first semester, the focus has been on developing the Energy Use Intensity (EUI) estimation technique, using globally available features to predict EUI. By selecting these key features, the goal has been to generate the first iteration of EUI predictions. The target for this stage is to achieve a Mean Absolute Percentage Error (MAPE) in the range of 30-40%. While this is the ideal range for this milestone, it is possible that we may not meet this target at this stage. Refining and improving this technique will be the focus for the second semester.

In the second semester, the objective will be to refine the model by incorporating additional features and enhancing its performance. The final goal is to enable global EUI prediction, providing a high-resolution, actionable method for estimating direct GHG emissions from building energy use.


## 3. Background <a name="Background"></a>

Existing estimates of anthropogenic CO2 emissions are provided by several sources, including the Open-source Data Inventory for Anthropogenic CO2 [5], the Community Emissions Data System [6], the Emissions Database for Global Atmospheric Research (EDGAR) [7], the Global Carbon Grid [8], and the Global Gridded Daily CO2 Emissions Dataset (GRACED) [9]. While GRACED data is updated near-monthly, most of the other key datasets have a production latency of a year or more. Furthermore, the highest resolution available across these datasets is 0.1 decimal degrees, which corresponds to approximately 11 km near the equator. Additionally, only a few of these models provide a breakdown of emissions into residential and commercial subsectors, or offer separate emissions estimates for individual greenhouse gases.

## 4. Data Description <a name="DataDescription"></a>

### 4.1 Building Emissions Estimation <a name="BuildingEmissionsEstimation"></a>

To estimate greenhouse gas (GHG) emissions from buildings, we will use Energy Use Intensity (EUI) as a central metric. EUI measures the energy consumption per square meter of building space, making it a valuable indicator for emissions estimation. By combining EUI values with total building floor area and an emissions factor, we can calculate the GHG emissions associated with buildings.

The estimation formula is:
![Formula](/figures/formula.png)

### 4.2 Feature Description <a name="FeatureDescription"></a>

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

## 5. Methods <a name="Methods"></a>

### 5.1 Feature Engineering <a name="FeatureEngineering"></a>
Feature engineering is essential to transform raw data into meaningful representations that enhance model performance and predictive accuracy. In this study, we applied the following techniques:  

1. **Heading Degree Days Calculation:**  
   Calculated using temperature data to derive features measure the demand for heating energy based on the difference between outdoor temperature and a baseline "comfort" temperature, typically 65°F (18°C).  

2. **Comfort Index Calculation:**  
   Derived using temperature and humidity data to quantify and evaluate human thermal comfort, which is influenced by environmental factors like temperature and humidity.

3. **Cross-Feature Interaction:**  
   Combined multiple features to create new interaction terms that capture relationships between variables.  

4. **Clustering Features:**  
   Apply clustering algorithms (e.g., k-means) to group data points and used cluster labels as additional features for modeling.

### 5.2 Nearest Reference Mapping <a name="NearestReferenceMapping"></a>

Nearest Reference Mapping involves assign each data point to its closest reference location based on a defined distance metric, enriching the dataset with relevant features from these reference points. 

In this project, we aim to assigning **EUI values** to each data point based on its nearest starting point with known ground truth. By using the EUI values as features and incorporating spatial context into our model, we aim to improve the model’s starting point and enhance prediction accuracy for global projections. 

### 5.3 Supervised Machine Learning <a name="SupervisedMachineLearning"></a>  


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

# 
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

As part of the initial iterations of the model, the following results were obtained using Linear Regression and KNN. These models serve as baseline models, providing a starting point for future improvements and model refinements.

#### Linear Regression

| Target                 | Strategy       | Model             | MSE  | R²     | MAE   | RMSE  | MAPE   | WAPE   |
|------------------------|----------------|-------------------|------|--------|-------|-------|--------|--------|
| Non-residential EUI   | Within-Domain  | Linear Regression | 3635 | -0.90  | 36.53 | 47.15 | 19.88  | 16.94  |
| Non-residential EUI   | Cross-Domain   | Linear Regression | 4238 | -12.85 | 50.06 | 60.41 | 31.89  | 31.10  |
| Non-residential EUI   | All-Domain     | Linear Regression | 3531 | -2.09  | 40.34 | 47.89 | 21.20  | 20.24  |
| Residential EUI       | Within-Domain  | Linear Regression | 1811 | -0.09  | 24.09 | 33.87 | 20.62  | 17.97  |
| Residential EUI       | Cross-Domain   | Linear Regression | 2613 | -1.92  | 40.81 | 48.45 | 40.59  | 34.98  |
| Residential EUI       | All-Domain     | Linear Regression | 1486 | -0.98  | 28.58 | 35.90 | 25.75  | 23.31  |

#### KNN

| Target                 | Strategy       | Model | MSE  | R²     | MAE   | RMSE  | MAPE   | WAPE   |
|------------------------|----------------|-------|------|--------|-------|-------|--------|--------|
| Non-residential EUI   | Within-Domain  | KNN   | 1437 | -0.07  | 21.12 | 31.00 | 10.36  | 10.09  |
| Non-residential EUI   | Cross-Domain   | KNN   | 3469 | -10.48 | 35.51 | 52.19 | 20.88  | 20.57  |
| Non-residential EUI   | All-Domain     | KNN   | 1290 | 0.12   | 18.94 | 29.86 | 9.56   | 9.26   |
| Residential EUI       | Within-Domain  | KNN   | 1044 | 0.32   | 20.22 | 27.75 | 15.40  | 15.37  |
| Residential EUI       | Cross-Domain   | KNN   | 2493 | -1.74  | 36.94 | 47.34 | 31.88  | 30.87  |
| Residential EUI       | All-Domain     | KNN   | 1042 | 0.30   | 20.14 | 27.35 | 15.31  | 15.37  |



## 9. Repository Structure and Usage <a name="RepositoryStructureAndUsage "></a>
This section provides an overview of the repository's structure, explaining the purpose of each directory and file. It also includes instructions for navigating and using the code.

### Directory Structure

```python

├── LICENSE
├── README.md
├── data
│   ├── 01_raw
│   │   ├── HDI_educationalIndex_incomeIndex.csv
│   │   ├── gdp_data.csv
│   │   └── population.csv
│   ├── 02_interim
│   │   ├── CDD.csv
│   │   └── HDD.csv
│   └── 03_processed
│       ├── merged_df.csv
│       └── merged_df_HDD.csv
├── figures
│   ├── diagram.png
│   ├── experimental_design.png
│   ├── feature_importance.png
│   ├── formula.png
│   └── region_map.png
├── notebooks
│   ├── 01_DataPreprocessing.ipynb
│   ├── 02_HDDPreprocessing.ipynb
│   ├── 03_HumidityPreprocessing.ipynb
│   ├── 04_FeatureImportance.ipynb
│   ├── 05_Experiments.ipynb
│   ├── 06_Model.ipynb
│   └── 07_Plots.ipynb
├── results
│   ├── results_20241114_1231_all_domain_knn.csv
│   ├── results_20241114_1231_all_domain_lr.csv
│   ├── results_20241114_1231_cross_domain_knn.csv
│   ├── results_20241114_1231_cross_domain_lr.csv
│   ├── results_20241114_1231_total_knn.csv
│   ├── results_20241114_1231_total_lr.csv
│   ├── results_20241114_1231_within_domain_knn.csv
│   └── results_20241114_1231_within_domain_lr.csv
└── slide_decks
    └── Climate_TRACE_Presentation.pdf



```


---


1. **`data/`**  
   - Contains all datasets used in the project. It is organized into subfolders:
     - `01_raw/`: Raw, unprocessed datasets like HDI, GDP, and population data.  
     - `02_interim/`: Intermediate processed files such as HDD and CDD values.  
     - `03_processed/`: Fully processed datasets ready for modeling (e.g., `merged_df.csv`).  

2. **`figures/`**  
   - Contains visual resources such as diagrams, maps, and other illustrations used in presentations and documentation.  

3. **`notebooks/`**  
   - Jupyter notebooks used for data processing, feature engineering, modeling, and analysis. Notebooks are ordered and labeled for clarity:  
     - **01_DataPreprocessing.ipynb**: Combines raw data into a single dataset.  
     - **02_HDDProcessing.ipynb**: Calculates Heating Degree Days (HDD) using global temperature data.  
     - **03_HumidityProcessing.ipynb**: Computes humidity values.  
     - **04_FeatureImportance.ipynb**: Analyzes feature importance.  
     - **05_Experiments.ipynb**: Contains experimental setups and evaluations.  
     - **06_Model.ipynb**: Implements machine learning models and saves results.  
     - **07_Plots.ipynb**: Generates visualizations for analysis and reporting.  

4. **`results/`**  
   - Stores evaluation outputs from various modeling strategies (e.g., `all_domain` or `cross_domain`) and models (e.g., KNN, Logistic Regression).  

5. **`README.md`**  
   - The entry point of the repository, providing an overview, key results, and links to all major components.  



#### Usage Instructions  

1. **Setup**:  
   Clone the repository and ensure all dependencies are installed. Use `requirements.txt` if available.  

2. **Data Processing**:  
   - Start with `01_DataPreprocessing.ipynb` to merge raw datasets.  
   - Use `02_HDDProcessing.ipynb` and `03_HumidityProcessing.ipynb` to compute additional features.  

3. **Modeling**:  
   - Open `06_Model.ipynb` to train models and evaluate performance across domains.  

4. **Results Analysis**:  
   - Use the `results/` directory to analyze model outputs and metrics.  

5. **Figures and Visuals**:  
   - All generated plots and diagrams are stored in `figures/` for easy reference in presentations or reports.  



## 10. Resources  <a name="Resources"></a>
1. [Greet Janssens-Maenhout, Monica Crippa, Diego Guizzardi, Marilena Muntean, Edwin Schaaf, Frank Dentener, Peter Bergamaschi, Valerio Pagliari, Jos G. J. Olivier, Jeroen A. H. W. Peters, John A. van Aardenne, Suvi Monni, Ulrike Doering, A. M. Roxana Petrescu, Efisio Solazzo, and Gabriel D. Oreggioni. (July 2019). EDGARv4.3.2 Global Atlas of the three major greenhouse gas emissions for the period 1970–2012.](https://essd.copernicus.org/articles/11/959/2019/)
2. [Markakis, P. J., Gowdy, T., Malof, J. M., Collins, L., Davitt, A., Volpato, G., & Bradbury, K. (2023). High-resolution global building emissions estimation using satellite imagery.](https://www.climatechange.ai/papers/neurips2023/128/paper.pdf)
3. [Marshall Burke*, Anne Driscoll, David B. Lobell, Stefano Ermon. (2021). Using satellite imagery to understand and promote sustainable development.](https://www.science.org/doi/full/10.1126/science.abe8628)
4. [Xinyu Dou, Yilong Wang, Philippe Ciais, Frédéric Chevallier, Steven J. Davis, Monica Crippa, Greet Janssens-Maenhout, Diego Guizzardi, Efisio Solazzo, Feifan Yan, Da Huo, Bo Zheng, Biqing Zhu, Duo Cui, Piyu Ke, Taochun Sun, Hengqi Wang, Qiang Zhang, Pierre Gentine, Zhu Deng, and Zhu Liu. (2022). Near-realtime global gridded daily CO2 emissions.](https://www.sciencedirect.com/science/article/pii/S2666675821001077)
5. [Oda, T., Maksyutov, S., & Andres, R. J. (2018). The Open-source Data Inventory for Anthropogenic CO2, version 2016 (ODIAC2016): A global monthly fossil fuel CO2 gridded emissions data product for tracer transport simulations and surface flux inversions. Earth System Science Data, 10(1), 87–107.](https://doi.org/10.5194/essd-10-87-2018)
6. [McDuffie, E. E., Smith, S. J., O’Rourke, P., Tibrewal, K., Venkataraman, C., Marais, E. A., Zheng, B., Crippa, M., Brauer, M., & Martin, R. V. (2020). A global anthropogenic emission inventory of atmospheric pollutants from sector- and fuel-specific sources (1970–2017): An application of the Community Emissions Data System (CEDS). Earth System Science Data, 12(4), 3413–3442.](https://doi.org/10.5194/essd-12-3413-2020)
7. [Janssens-Maenhout, G., Crippa, M., Guizzardi, D., Muntean, M., Schaaf, E., Dentener, F., Bergamaschi, P., Pagliari, V., Olivier, J. G. J., Peters, J. A. H. W., van Aardenne, J. A., Monni, S., Doering, U., Petrescu, A. M. R., Solazzo, E., & Oreggioni, G. D. (2019). EDGAR v4.3.2 Global Atlas of the three major greenhouse gas emissions for the period 1970–2012. Earth System Science Data, 11(3), 959–1002.](https://doi.org/10.5194/essd-11-959-2019)
8. [Tong, D., Zhang, Q., Davis, S. J., Liu, F., Zheng, B., Geng, G., Xue, T., Li, M., Hong, C., Lu, Z., Streets, D. G., Guan, D., & He, K. (2018). Targeted emission reductions from global super-polluting power plant units. Nature Sustainability, 1(1), 59–68.](https://doi.org/10.1038/s41893-017-0003-y)
9. [Dou, X., Wang, Y., Ciais, P., Chevallier, F., Davis, S. J., Crippa, M., Janssens-Maenhout, G., Guizzardi, D., Solazzo, E., Yan, F., Huo, D., Zheng, B., Zhu, B., Cui, D., Ke, P., Sun, T., Wang, H., Zhang, Q., Gentine, P., Deng, Z., & Liu, Z. (2022). Near-realtime global gridded daily CO2 emissions. The Innovation, 3(1), 100182.](https://doi.org/10.1016/j.xinn.2021.100182)




## 11. Contributors  <a name="Contributors"></a>
[Jiechen Li](https://github.com/carrieli15)  
[Meixiang Du](https://github.com/dumeixiang)  
[Yulei Xia](https://github.com/AliciaXia222)  
[Barbara Flores](https://github.com/BarbaraPFloresRios)  



#### Project Mentor and Client: [Dr. Kyle Bradbury](https://energy.duke.edu/about/staff/kyle-bradbury)


