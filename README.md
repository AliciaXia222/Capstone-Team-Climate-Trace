# 🌎 Estimating Global Greenhouse Gas Emissions from Buildings

## Table of Contents

1. [Abstract](#Abstract)
2. [Introduction](#Introduction)  
   2.1 [Project Motivation](#ProjectMotivation)  
   2.2 [Problem Statement](#ProblemStatement)  
   2.3 [Goals](#Goals)  
   2.4 [Project Deliverables and Presentation Materials](#ProjectDeliverablesAndPresentationMaterials)   
3. [Background](#Background)  
4. [Data and Features Overview](#DataAndFeaturesOverview)  
   4.1 [GHG Estimation](#GHGEstimation)  
   4.2 [Features for EUI Prediction](#FeaturesForEUIPrediction)  
   4.3 [Generated data](#GeneratedData)  
5. [Methods](#Methods)  
   5.1 [Feature Engineering](#FeatureEngineering)  
   5.2 [Nearest Reference Mapping](#NearestReferenceMapping)  
   5.3 [Supervised Machine Learning](#SupervisedMachineLearning)  
   5.4 [Experimental Design](#ExperimentalDesign)
7. [Experiments](#Experiments)  
   6.1 [Feature Importance](#FeatureImportance)  
   6.2 [Models](#Models)
8. [Conclusion](#Conclusion)  
9. [Repository Structure and Usage](#RepositoryStructureAndUsage)
10. [Resources](#Resources)
11. [Contributors](#Contributors) 


## 1. Abstract <a name="Abstract"></a>

This project develops a machine learning model to estimate direct greenhouse gas (GHG) emissions from residential and non-residential building energy consumption. The model predicts energy use intensity (EUI) by incorporating climatic, geographical, and socioeconomic variables for both residential and non-residential buildings. These EUI estimates, along with global building floor area, will be used in the next stage of this project to calculate direct GHG emissions from buildings, offering a timely, high-resolution method for global emissions estimation.

This current work outlines preliminary EUI estimation techniques, with the primary focus on minimizing the Mean Absolute Percentage Error (MAPE) with a target range of less than 30-40%. Other performance metrics, such as R², Mean Squared Error (MSE), Mean Absolute Error (MAE), and Weighted Absolute Percentage Error (WAPE), were also considered. Future iterations will refine and expand the model by incorporating additional features to enhance performance, ultimately addressing the challenge of estimating global direct GHG emissions from buildings.

Analysis showed that ensemble and tree-based models, including Random Forest, XGBoost, and CatBoost, outperformed traditional methods, with Random Forest achieving the best results overall and a MAPE averaging less than 21% across different cross-validation strategies. However, R² was generally low across all models. Additionally, results varied by region, with models performing better within-domain but struggling to generalize effectively in cross-domain scenarios.

## 2. Introduction <a name="Introduction"></a>

### 2.1  Project Motivation  <a name="ProjectMotivation"></a>

Global warming is one of the most critical challenges of our time, and to address it effectively, we need more detailed information on where and when greenhouse gas emissions occur. This data is crucial for setting actionable emissions reduction goals and enabling policymakers to make informed decisions. Given this situation, Climate TRACE, a non-profit coalition of organizations, is building a timely, open, and accessible inventory of global emissions sources, currently covering around 83% of global emissions.

Building direct emissions are responsible for between 6% and 9% of global GHG emissions, primarily due to onsite fossil fuel combustion for heating, water heating, and cooking. Indirect emissions from lighting, consumer electronics, and air conditioning are excluded, as they are typically electric and accounted for separately in the Climate TRACE database.

Despite their significant contribution to global emissions, the building sector still lacks the timely, high-resolution, and low-latency data needed to assess GHG emissions accurately. Current methodologies rely on outdated data, often delayed by over a year, or on self-reported data that is scarce or unavailable globally.

### 2.2  Problem Statement  <a name="ProblemStatement"></a>

Specifically, we can define our problem statement as follows:

***The building sector lacks timely, high-resolution data on direct greenhouse gas (GHG) emissions, limiting the ability to accurately track and reduce emissions from building energy use.***

### 2.3  Goals  <a name="Goals"></a>

The goal of this project is to develop a machine learning model to estimate greenhouse gas (GHG) emissions based on building energy consumption. The model will predict energy use intensity (EUI) using climatic, geographical, and socioeconomic variables. These EUI estimates, along with building area data, will be used to calculate direct GHG building emissions.

In the first semester, the focus has been on developing the Energy Use Intensity (EUI) estimation technique, using globally available features to predict EUI. By selecting these key features, the goal has been to generate the first iteration of EUI predictions. The target for this stage is to achieve a Mean Absolute Percentage Error (MAPE) in the range of 30-40%. While this is the ideal range for this milestone, it is possible that we may not meet this target at this stage. Refining and improving this technique will be the focus for the second semester.

In the second semester, the objective will be to refine the model by incorporating additional features and enhancing its performance. The final goal is to enable global EUI prediction, providing a high-resolution, actionable method for estimating direct GHG emissions from building energy use.

### 2.4 Project Deliverables and Presentation Materials <a name="ProjectDeliverablesAndPresentationMaterials"></a>

This section provides an overview of the key deliverables and presentation materials developed throughout the project. These materials summarize the project's progress, next steps, and areas to explore in the upcoming semester, offering insights into the work completed and the outcomes achieved.

1. The deliverables we've agreed to provide to our client this semester can be found [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/tree/main/deliverables_agreement).

2. For a visual summary of the project, check out the slide deck presentation [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/slide_decks/Climate_TRACE_Presentation.pdf).

3. The mid-point slide deck of analysis and results can be found [here](https://docs.google.com/presentation/d/1aeell_KmJwJAF3aopTo4ghbe1blzv2hjZ7Oo8uiPV3E/edit?usp=sharing).

## 3. Background <a name="Background"></a>

The accurate estimation of anthropogenic CO2 emissions is critical for understanding global climate change and formulating effective policies. Existing estimates are provided by several key datasets, including the Open-source Data Inventory for Anthropogenic CO2 ([Oda, Maksyutov, & Andres, 2018](#oda2018)), the Community Emissions Data System ([McDuffie et al., 2020](#mcduffie2020)), the Emissions Database for Global Atmospheric Research (EDGAR) ([Janssens-Maenhout et al., 2019](#janssens2019)), the Global Carbon Grid ([Tong et al., 2018](#tong2018)), and the Global Gridded Daily CO2 Emissions Dataset ([Dou et al., 2022](#dou2022)). While the GRACED data is updated nearly monthly, most of the other key datasets suffer from a significant production latency, often of a year or more. Additionally, the highest resolution available across these datasets is 0.1 decimal degrees, roughly equivalent to an 11 km grid near the equator. Furthermore, only a few of these datasets provide detailed breakdowns of emissions by sector, such as residential and commercial subsectors, or offer separate estimates for different greenhouse gases([Markakis et al., 2023](#markakis2023)).

In response to these challenges, recent advancements have been made in the development of more granular and timely emissions estimation methods. One such breakthrough is the High-resolution Global Building Emissions Estimation using Satellite Imagery model by [Markakis et al. (2023)](#markakis2023). This innovative model offers high-resolution, global emissions estimates for both residential and commercial buildings at a 1 km² resolution, with updates on a monthly basis. By leveraging satellite imagery-derived features and machine learning techniques, the model estimates direct emissions from buildings. This approach addresses the temporal and spatial limitations of previous datasets by predicting building areas, estimating energy use intensity, and calculating emissions based on regional fuel mixes. Unlike other datasets like GRACED and EDGAR, this model offers more granular insights into emissions at a higher frequency and resolution, making it a crucial tool for policymakers working to reduce emissions in the building sector on a global scale.


## 4. Data and Features Overview <a name="DataAndFeaturesOverview"></a>

### 4.1 GHG Estimation <a name="GHGEstimation"></a>

To estimate greenhouse gas (GHG) emissions from buildings, we will use Energy Use Intensity (EUI) as a central metric. EUI measures the energy consumption per square meter of building space, making it a valuable indicator for emissions estimation. By combining EUI values with total building floor area and an emissions factor, we can calculate the GHG emissions associated with buildings.

The estimation formula is:
![Formula](/figures/01_formula.png)

### 4.2 Features for EUI Prediction <a name="FeaturesForEUIPrediction"></a>

In this section, we describe both the dependent variable of our model (EUI) and the independent features we are exploring to predict Energy Use Intensity (EUI) in buildings. The independent features include factors that are considered potentially influential on energy consumption, based on both prior research and discussions with experts in the field. These independent features serve as inputs to the model, and some of them are used to calculate additional derived features, such as the Heating Degree Days (HDD) and Comfort Index, which are explained further in the Feature Engineering section. Below, we outline the open datasets we are using to build and refine our EUI prediction model.

1. **[EUI](https://drive.google.com/uc?id=12qGq_DLefI1RihIF_RKQUyJtm480-xRC)**:
EUI is a metric used to measure the intensity of energy use in buildings. These EUI values serve as our dependent variable, or the target we seek to predict, in our model. This dataset, provided by the client, contains 482 entries and focuses on two key variables:

   - Residential EUI:  Indicates the energy consumption of residential buildings, expressed in kWh/m²/year.
   - Non-Residential EUI: Reflects the energy consumption of non-residential buildings, also expressed in kWh/m²/year.
  
To better understand the distribution of this variable, we can observe the following map, which visualizes how EUI is distributed across the different regions and building types.

![EUI map](/figures/02_eui_map.png)

2. **[Temperature](https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics?tab=overview)**: Air temperature at 2m above the surface, interpolated using atmospheric conditions. Measured in kelvin. This feature is essential for estimating heating needs (which contribute to direct energy use in buildings) and is later used to calculate Heating Degree Days (HDD) and Cooling Degree Days (CDD).

3. **[Dewpoint Temperature](https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics?tab=overview)**: The temperature at which air at 2m above the surface becomes saturated, indicating humidity levels. Measured in kelvin. 

4. **[Latitude](https://download.geonames.org/export/dump/)**: Provides global latitude data in decimal degrees (WGS84 coordinate reference system), adding geographical context to our analysis.

5. **[Longitude](https://download.geonames.org/export/dump/)**: Provides global longitude data in decimal degrees (WGS84 coordinate reference system), complementing the latitude data for geographical analysis.

6. **[Population](https://globaldatalab.org/shdi/metadata/pop/)**: Includes population data for various countries and regions from 1960 to 2023. For our analysis, we extracted the population figures for 2023 to align with our project goals.

7. **[GDP](https://globaldatalab.org/shdi/metadata/shdi/)**: Contains data on human development, health, education, and income across 160+ countries from 1990 to 2022. We used the GDP values for 2022 as a key feature for our model.

8. **[Human Development Index (HDI)](https://globaldatalab.org/shdi/metadata/shdi/)**: HDI measures a country's achievements in three key areas:  
   - *Health*: A long and healthy life.  
   - *Knowledge*: Access to education.  
   - *Standard of Living*: A decent standard of living.  
   We extracted data for the year 2022 to maintain consistency with other datasets.

9. **[Urbanization Rate](https://data.worldbank.org/indicator/SP.URB.TOTL.IN.ZS?end=2023&start=2023&view=map&year=2022)**: Urbanization rate reflects the average annual growth of urban populations. For consistency, we used data from 2022.

10. **[Educational Index](https://globaldatalab.org/shdi/metadata/edindex/)**: This index comprises two indicators:  
   - *Mean Years of Schooling (MYS)*: The average years of schooling for adults aged 25 and above.  
   - *Expected Years of Schooling (EYS)*: The anticipated years of education for the current population.  

11. **[Paris Agreement](https://unfccc.int/process-and-meetings/the-paris-agreement)**: The Paris Agreement is an international treaty adopted by 196 parties in 2015. We used this information to create a binary variable to indicate whether a country is a signatory.


### 4.3 Generated Data <a name="GeneratedData"></a>
After feature engineering and merging our datasets, we've generated the final dataset for model input, containing 482 data points. It can be accessed [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/data/03_processed/merged_df.csv)

## 5. Methods <a name="Methods"></a>

### 5.1 Feature Engineering <a name="FeatureEngineering"></a>
Feature engineering is essential to transform raw data into meaningful representations that enhance model performance and predictive accuracy. In this study, we applied the following techniques:  

1. **Heating Degree Days Calculation:**  
   Calculated using temperature data to derive features measure the demand for heating energy based on the difference between outdoor temperature and a baseline "comfort" temperature, typically 65°F (18°C).
   
2. **Cooling Degree Days Calculation:**  
   Calculated using temperature data to derive features measure the demand for Cooling related energy usage based on the difference between outdoor temperature and a baseline "comfort" temperature, typically 65°F (18°C). 

3. **GDP per Capita Calculation:**
We use GDP per capita, which is the result of dividing total GDP by the population, as it provides more relevant information for our model. This approach better captures the economic impact on energy consumption at the individual level, enabling more accurate comparisons across regions with varying population sizes. 

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

   - **Random Forest:**
     Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. It is particularly useful for handling high-dimensional data and capturing complex, non-linear relationships.
   - **XGBoost:**  
     XGBoost is an optimized gradient boosting algorithm that performs well in a variety of prediction tasks. It builds an ensemble of decision trees sequentially, improving the model’s performance by reducing bias and variance.
   
   - **CatBoost:**  
     CatBoost is another gradient boosting algorithm known for its handling of categorical features without the need for explicit preprocessing. It is expected to provide competitive results, particularly in datasets with mixed types of variables.

The combination of linear models, distance-based methods like KNN, and powerful ensemble models like XGBoost and CatBoost will allow us to capture a range of patterns in the data, from simple linear trends to more complex interactions and non-linear relationships.

### 5.4 Experimental Design <a name="ExperimentalDesign"></a>
Given the challenge of regional variations in global data, we will validate our predictions at the regional level across five distinct regions using three strategies to identify biases and improve model robustness. The regions we are using are defined as follows:

Given the challenge of regional variations in global data, we will validate our predictions at the regional level across five distinct regions using three strategies to identify biases and improve model robustness. This approach helps to account for local differences in energy use patterns and improve the model’s predictive accuracy across diverse contexts. The regions we are using are defined as follows:

1. Asia & Oceania
2. Europe
3. Africa
4. Central and South America
5. Northern America

The data points in our dataset, which we intend to predict, are distributed across these various regions, as illustrated in the following map.


![Geographic Distribution of Data Points by Region](/figures/03_region_map.png)

The strategies we will be using are as follows:

![Image](/figures/04_experimental_design.png)

We aim to assess our model's generalization by comparing its performance within the same region (Within-Domain) and its ability to extrapolate to other regions (Cross-Domain). The goal is to reduce the gap between these strategies to improve accuracy and understand extrapolation errors. Additionally, we want to understand if there are regions that perform better than others in specific outcomes, which can help us tailor our model to regional differences.

## 6. Experiments <a name="Experiments"></a>


### 6.1 Feature Importance <a name="FeatureImportance"></a>

To find the most important factors in building energy use and greenhouse gas emissions, we used a linear regression model. The target variable, energy use intensity (EUI), was calculated as the total of residential and non-residential energy use (kWh/m²/year).

The model included factors like GDP per capita, urbanization rate, latitude, and subnational HDI. To make all variables comparable, we standardized the data before training the model. Heating Degree Days (HDD), which measures heating demand based on temperature, turned out to be the most important factor, showing how much temperature affects energy use.

In the future, the model could include other temperature-related factors, like average temperature and humidity, which were not included in this iteration. For details on the calculations, check the [Feature Importance Notebook](/notebooks/050_FeatureImportance.ipynb).   

![Feature Importance](/figures/05_feature_importance.png)

### 6.2 Models <a name="Models"></a>

#### Comparison Between Models

In this section, we evaluate the performance of several machine learning models used for predicting Energy Use Intensity (EUI) and estimating greenhouse gas (GHG) emissions from buildings. The models tested include Linear Regression (LR), Linear Regression with Lasso and Ridge regularization, K-Nearest Neighbors (KNN), Random Forest, XGBoost, and CatBoost. The evaluation metrics, such as Mean Absolute Percentage Error (MAPE) and R², are used to assess model performance across different feature sets. The models are also evaluated across various cross-validation strategies to ensure robustness and generalizability. The specific features utilized in each model, along with the hyperparameters tested, can be found in detail in the tables [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/tree/main/results) and are summarized in this [table](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/results/comparison_average_results.csv). The findings and recommendations for model improvement will be further explored in the [Results & Analysis slide deck](https://docs.google.com/presentation/d/1aeell_KmJwJAF3aopTo4ghbe1blzv2hjZ7Oo8uiPV3E/edit?usp=sharing).


The following graphs display the average performance metrics for MAPE, R², and RMSE for both residential and non-residential buildings across different cross-validation strategies: within domain, cross-domain, and all domain. These averages are calculated for various regions, helping us to identify the best model and select our EUI Estimation Technique. By analyzing these results, we can determine which model performs best across different scenarios, guiding our final decision on the most effective approach for predicting EUI. In addition to MAPE, R², and RMSE, other evaluation metrics can be found [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/results/comparison_average_results.csv).

   - **MAPE**  
   The MAPE across validation strategies revealed that predictions for non-residential buildings generally demonstrated superior predictive capability compared to residential buildings. Ensemble models (Random Forest, XGBoost, and CatBoost) consistently outperformed traditional approaches. We can observe that the within-domain strategy delivered better results, while the cross-domain strategy presented more challenges, with the all-domain strategy serving as an intermediate point. One of the challenges that arises is how to improve generalization in order to reduce the gap between these strategies, ultimately enhancing model performance across different regions. On average, we achieved the target of a MAPE below 30-40%, although it is important to note that some individual predictions fell outside this range, as will be discussed further later.

![eui_predictions_all_domain](/figures/model_plots/00_model_comparison_mape.png)


   - **R²**  
   The R² analysis reveals significant challenges in model performance across different strategies. For within-domain validation, ensemble models showed moderate positive R² values (0.22-0.52), with Random Forest achieving the best performance. However, cross-domain validation resulted in consistently negative R² values across all models. This suggests that the relationships between features and EUI are highly complex and region-dependent. The all-domain strategy showed intermediate results, with ensemble models maintaining slightly positive R² values (0.11-0.47) while linear models continued to perform poorly. These results, especially the negative R² values, indicate that the relationship between our features and EUI is strongly non-linear and varies significantly across different geographical regions, highlighting the importance of using more sophisticated modeling approaches and potentially developing region-specific models.

![eui_predictions_all_domain](/figures/model_plots/00_model_comparison_r2.png)

   - **RMSE** 
   The RMSE analysis further supports the patterns observed in previous metrics. Within-domain, ensemble models achieved the lowest errors. Cross-domain validation revealed substantially higher errors, particularly for linear models (RMSE ranging from 59.4 to 62.6 for non-residential). The magnitude of these errors decreased in all-domain validation, though ensemble models maintained their superior performance. 

![eui_predictions_all_domain](/figures/model_plots/00_model_comparison_rmse.png)

#### Best Model Overall: Random Forest

Based on our evaluation across metrics, we selected Random Forest as our primary model for EUI prediction. While some models occasionally outperformed in specific scenarios, Random Forest demonstrated the most consistent and balanced performance across validation strategies and building types. It achieved MAPE values below 15% for non-residential and 21% for residential in within-domain validation, maintained positive R² values (0.22-0.52 within-domain), and showed stable RMSE values (29.3 non-residential, 23.6 residential within-domain). This consistent performance, along with its ability to handle non-linear relationships and maintain stability in cross-domain scenarios, makes Random Forest the most reliable choice for global EUI prediction.

The following figure shows detailed performance metrics for the Random Forest model across different validation strategies and building types. Detailed results by region can be found in this [table](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/results/20241208_2046_rf_detailed_results.csv), while average performance metrics are available [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/results/20241208_2046_rf_average_results.csv).

![eui_predictions_all_domain](/figures/06_avg_rf.png)

A detailed analysis of the Random Forest model's performance revealed distinct patterns across building types and validation strategies. For non-residential buildings, the model achieved its best performance in within-domain validation with a MAPE of 8.96% and R² of 0.22, though performance declined in cross-domain scenarios (MAPE 13.58%, R² -0.36). For residential buildings, while showing higher error rates (MAPE 12.76% within-domain), it demonstrated stronger explanatory power (R² 0.52). The all-domain strategy provided a balanced middle ground, with MAPE of 9.98% and 13.08% for non-residential and residential buildings respectively. These results demonstrate that while geographical variations impact model performance, the Random Forest consistently maintains error levels well within our target range of 30-40% MAPE across all scenarios, making it a robust choice for global EUI prediction.

The following figures display the prediction performance across different regions for both residential (top row) and non-residential (bottom row) EUI. Each scatter plot compares predicted versus actual values

1. **Within Domain**:  

   - **Actual EUI vs. Predicted EUI**  
![eui_predictions_all_domain](/figures/model_plots/rf_within_domain_eui_predictions.png)

   - **Error Distribution Plot**  
![eui_predictions_all_domain](/figures/model_plots/rf_within_domain_error_distribution.png)

2. **Cross Domain**:  

   - **Actual EUI vs. Predicted EUI** 
![eui_predictions_cross_domain](/figures/model_plots/rf_cross_domain_eui_predictions.png)

   - **Error Distribution Plot**  
![eui_predictions_all_domain](/figures/model_plots/rf_cross_domain_error_distribution.png)


3. **All Domain**:  

   - **Actual EUI vs. Predicted EUI** 
![eui_predictions_all_domain](/figures/model_plots/rf_all_domain_eui_predictions.png)

   - **Error Distribution Plot**  
![eui_predictions_all_domain](/figures/model_plots/rf_all_domain_error_distribution.png)

## 7. Conclusion <a name="Conclusion"></a>


Analysis showed that ensemble and tree-based models, including Random Forest, XGBoost, and CatBoost, outperformed traditional methods, with Random Forest achieving the best results overall, with a MAPE averaging less than 25% in various scenarios. However, R² was generally low across all models. Additionally, results varied by region, with models performing better within-domain but struggling to generalize effectively in cross-domain scenarios.

Next steps could include evaluating additional variables, such as weather-related variables and even satellite image data, to further improve the model's performance and robustness.


## 8. Repository Structure and Usage <a name="RepositoryStructureAndUsage"></a>
This section provides an overview of the repository's structure, explaining the purpose of each directory and file. It also includes instructions for navigating and using the code.

### Directory Structure

```python
.
├── LICENSE
├── README.md
├── data
│   ├── 01_raw
│   │   ├── HDI_educationalIndex_incomeIndex.csv
│   │   ├── gdp_data.csv
│   │   └── population.csv
│   ├── 02_interim
│   │   ├── CDD.csv
│   │   ├── HDD.csv
│   │   └── Humidity.csv
│   └── 03_processed
│       └── merged_df.csv
├── deliverables_agreement
│   └── Mid-Point Deliverables - Climate Trace.pdf
├── figures
│   ├── 01_formula.png
│   ├── 02_eui_map.png
│   ├── 03_region_map.png
│   ├── 04_experimental_design.png
│   ├── 05_feature_importance.png
│   └── model_plots
├── notebooks
│   ├── 010_Download_WeatherData_API.ipynb
│   ├── 020_WeatherData_Preprocessing.ipynb
│   ├── 021_HumidityPreprocessing.ipynb
│   ├── 023_HDDPreprocessing.ipynb
│   ├── 030_DataPreprocessing.ipynb
│   ├── 040_Plots.ipynb
│   ├── 050_FeatureImportance.ipynb
│   ├── 060_Experiments_LR.ipynb
│   ├── 061_Experiments_KNN.ipynb
│   ├── 062_Experiments_RF.ipynb
│   ├── 063_Experiments_XGBoost.ipynb
│   ├── 064_Experiments_CatBoost.ipynb
│   ├── 070_Model_Comparison.ipynb
├── requirements.txt
├── results
├── slide_decks
│   └── Climate_TRACE_Presentation.pdf
└── src
    ├── __init__.py
    ├── __pycache__
    │   └── lib.cpython-311.pyc
    └── lib.py

```


1. **`data/`**  
   - Contains all datasets used in the project. It is organized into subfolders:
     - **01_raw/**: Raw, unprocessed datasets like HDI, GDP, and population data.  
     - **02_interim/**: Intermediate processed files such as HDD and CDD values.  
     - **03_processed/**: Fully processed datasets ready for modeling (e.g., merged_df.csv).  

2. **`figures/`**  
   - Contains visual resources such as diagrams, maps, and other illustrations used in presentations and documentation.  

3. **`notebooks/`**  

   - Jupyter notebooks used for data processing, feature engineering, modeling, and analysis. Notebooks are ordered and labeled for clarity:  
     - **010_Download_WeatherData_API.ipynb**: Downloads weather data from the Copernicus Climate Data Store (ERA5-Land daily statistics).  
     - **020_WeatherData_Preprocessing.ipynb**: Preprocesses weather data for model input.  
     - **023_HDDPreprocessing.ipynb**: Prepares Heating Degree Days (HDD) data.  
     - **030_DataPreprocessing.ipynb**: Prepares the final dataset for model input.  
     - **040_Plots.ipynb**: Generates visualizations for analysis and reporting.  
     - **050_FeatureImportance.ipynb**: Analyzes feature importance for model evaluation.  
     - **060_Experiments_LR.ipynb**: Sets up and evaluates experiments using Logistic Regression.  
     - **061_Experiments_KNN.ipynb**: Implements and evaluates K-Nearest Neighbors (KNN) models.  
     - **062_Experiments_RF.ipynb**: Runs experiments using Random Forest (RF).  
     - **063_Experiments_XGBoost.ipynb**: Executes XGBoost models for performance comparison.  
     - **064_Experiments_CatBoost.ipynb**: Configures and evaluates CatBoost models.  
     - **070_Model_Comparison.ipynb**: Compares the performance of different models across various datasets and variables.
5. **`results/`**  
   - Stores evaluation outputs from various modeling strategies (e.g., `all_domain` or `cross_domain`) and models (e.g., KNN, Logistic Regression).

6. **`src/`**  
   - Contains core Python scripts for the project.  
     - **lib.py**: Provides utility functions and shared modules for data preprocessing, feature extraction, and model evaluation, used across notebooks and scripts.  

7. **`requirements.txt`**  
   - Lists all dependencies needed for the project environment, ensuring reproducibility.  

8. **`README.md`**  
   - The entry point of the repository, providing an overview, key results, and links to all major components.  


### Usage Instructions  

1. **Setup**:  
   Clone the repository and ensure all dependencies are installed. Use `requirements.txt` 

2. **Data Processing**:  
   - Start with `01_DataPreprocessing.ipynb` to merge raw datasets.  
   - Use `02_HDDProcessing.ipynb` and `03_HumidityProcessing.ipynb` to compute additional features.  

3. **Modeling**:  
   - Open `06_Model.ipynb` to train models and evaluate performance across domains.  

4. **Results Analysis**:  
   - Use the `results/` directory to analyze model outputs and metrics.  

5. **Figures and Visuals**:  
   - All generated plots and diagrams are stored in `figures/` for easy reference in presentations or reports.  



## 9. Resources  <a name="Resources"></a>

1. [Dou, X., Wang, Y., Ciais, P., Chevallier, F., Davis, S. J., Crippa, M., Janssens-Maenhout, G., Guizzardi, D., Solazzo, E., Yan, F., Huo, D., Zheng, B., Zhu, B., Cui, D., Ke, P., Sun, T., Wang, H., Zhang, Q., Gentine, P., Deng, Z., & Liu, Z. (2022). Near-realtime global gridded daily CO2 emissions. *The Innovation, 3*(1), 100182.](https://doi.org/10.1016/j.xinn.2021.100182) <a name="dou2022"></a>

2. [Janssens-Maenhout, G., Crippa, M., Guizzardi, D., Muntean, M., Schaaf, E., Dentener, F., Bergamaschi, P., Pagliari, V., Olivier, J. G. J., Peters, J. A. H. W., van Aardenne, J. A., Monni, S., Doering, U., Petrescu, A. M. R., Solazzo, E., & Oreggioni, G. D. (2019). EDGAR v4.3.2 Global Atlas of the three major greenhouse gas emissions for the period 1970–2012. *Earth System Science Data, 11*(3), 959–1002.](https://doi.org/10.5194/essd-11-959-2019) <a name="janssens2019"></a>

3. [Markakis, P. J., Gowdy, T., Malof, J. M., Collins, L., Davitt, A., Volpato, G., & Bradbury, K. (2023). High-resolution global building emissions estimation using satellite imagery. *Climate Change AI*.](https://www.climatechange.ai/papers/neurips2023/128/paper.pdf) <a name="markakis2023"></a>

4. [McDuffie, E. E., Smith, S. J., O’Rourke, P., Tibrewal, K., Venkataraman, C., Marais, E. A., Zheng, B., Crippa, M., Brauer, M., & Martin, R. V. (2020). A global anthropogenic emission inventory of atmospheric pollutants from sector- and fuel-specific sources (1970–2017): An application of the Community Emissions Data System (CEDS). *Earth System Science Data, 12*(4), 3413–3442.](https://doi.org/10.5194/essd-12-3413-2020) <a name="mcduffie2020"></a>

5. [Oda, T., Maksyutov, S., & Andres, R. J. (2018). The Open-source Data Inventory for Anthropogenic CO2, version 2016 (ODIAC2016): A global monthly fossil fuel CO2 gridded emissions data product for tracer transport simulations and surface flux inversions. *Earth System Science Data, 10*(1), 87–107.](https://doi.org/10.5194/essd-10-87-2018) <a name="oda2018"></a>

6. [Tong, D., Zhang, Q., Davis, S. J., Liu, F., Zheng, B., Geng, G., Xue, T., Li, M., Hong, C., Lu, Z., Streets, D. G., Guan, D., & He, K. (2018). Targeted emission reductions from global super-polluting power plant units. *Nature Sustainability, 1*(1), 59–68.](https://doi.org/10.1038/s41893-017-0003-y) <a name="tong2018"></a>



## 10. Contributors  <a name="Contributors"></a>
[Jiechen Li](https://github.com/carrieli15)  
[Meixiang Du](https://github.com/dumeixiang)  
[Yulei Xia](https://github.com/AliciaXia222)  
[Barbara Flores](https://github.com/BarbaraPFloresRios)  


#### Project Mentor and Client: [Dr. Kyle Bradbury](https://energy.duke.edu/about/staff/kyle-bradbury)


