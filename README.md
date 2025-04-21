# 🌎 Estimating Global Greenhouse Gas Emissions from Buildings

## Table of Contents

1. [Abstract](#Abstract)
2. [Introduction](#Introduction)  
   2.1 [Project Motivation](#ProjectMotivation)  
   2.2 [Problem Statement](#ProblemStatement)  
   2.3 [GHG Estimation Formula](#GHGEstimation)    
   2.4 [Goals](#Goals)  
   2.5 [Project Deliverables and Presentation Materials](#ProjectDeliverablesAndPresentationMaterials)
3. [Background](#Background)
4. [Methodology](#Methodology)   
   4.1 [Overall Framework](#OverallFramework)  
   4.2 [Feature Map](#FeatureMap)   
   4.3 [Feature Engineering](#FeatureEngineering)
5. [Models](#Models)  
   5.1 [Supervised Machine Learning](#SupervisedMachineLearning)    
   5.2 [Experimental Design](#ExperimentalDesign)
6. [Experiments](#Experiments)    
   6.1 [Feature Importance](#FeatureImportance)    
   6.2 [Models](#Models)
7. [Conclusion](#Conclusion)
8. [Repository Structure and Usage](#RepositoryStructureAndUsage)
9. [Resources](#Resources)
10. [Contributors](#Contributors)   

## 1. Abstract <a name="Abstract"></a>

This project develops a machine learning model to estimate direct greenhouse gas (GHG) emissions from residential and non-residential building energy consumption. The model predicts energy use intensity (EUI) by incorporating climatic, geographical, and socioeconomic variables. These EUI estimates, combined with global building floor area, serve as a basis for calculating direct GHG emissions from buildings, offering a timely, high-resolution approach to global emissions estimation.

The work focuses on developing an EUI estimation technique, with an emphasis on minimizing the Mean Absolute Percentage Error (MAPE). Starting from a baseline K-Nearest Neighbors (KNN) model (K = 1), using only geographic location (latitude and longitude), with an average MAPE of 37.8% in cross domain validation, we reduced the error to an average of 17.51% using a Random Forest model and selecting the most important features among those evaluated. This represents a 54% improvement from the baseline in cross-domain validation, which is the most conservative strategy compared to all-domain and within-domain evaluations. This result highlights the robustness of the model and the effectiveness of the proposed approach.  

## 2. Introduction <a name="Introduction"></a>

### 2.1  Project Motivation  <a name="ProjectMotivation"></a>

In 2024, global carbon dioxide (CO₂) emissions reached a record 41.6 billion metric tons, equivalent to covering nearly 1.5 million football fields with a one-meter-thick layer of CO₂ ([Live Science, 2024](#livescience2024)). Global warming is one of the most critical challenges of our time, and to address it effectively, we need more detailed information on where and when GHG emissions occur. This data is crucial for setting actionable emissions reduction goals and enabling policymakers to make informed decisions. Given this situation, Climate TRACE, a non-profit coalition of organizations, is building a timely, open, and accessible inventory of global emissions sources that currently covers around 83% of global emissions.

Building direct emissions are responsible for between 6% and 9% of global GHG emissions, primarily due to onsite fossil fuel combustion for heating, water heating, and cooking. In contrast, indirect emissions from lighting, consumer electronics, and air conditioning are excluded as they typically result from electricity use and are accounted for separately in the Climate TRACE database.

Despite their significant contribution to global emissions, the building sector still lacks the timely, high-resolution, and low-latency data needed to assess GHG emissions accurately. Current methodologies rely on outdated or spatially limited data, and emissions inventories remain incomplete—52 countries lack emissions data after 2012, with even larger gaps at the subnational and local levels ([Climate TRACE, 2023](#climatetrace2023) ; [Luers et al., 2022](#luers2022)). In addition, available data is often spatially coarse or inconsistently reported, further limiting the effectiveness of climate action at local scales.

### 2.2  Problem Statement  <a name="ProblemStatement"></a>

Specifically, we can define our problem statement as follows:

***The building sector lacks timely, high-resolution data on direct greenhouse gas (GHG) emissions, limiting the ability to accurately track and reduce emissions from building energy use.***


### 2.3 GHG Estimation Formula <a name="GHGEstimation"></a>

To estimate GHG emissions from buildings, Energy Use Intensity (EUI) can be used as a central metric. EUI measures the energy consumption per square meter of building space, making it a valuable indicator for emissions estimation. By combining EUI values with total building floor area and an emissions factor, we can calculate the GHG emissions associated with buildings.


The estimation formula is:
![Formula](/figures/01_formula.png)
*<sub>Source: Markakis, P. J., Gowdy, T., Malof, J. M., Collins, L., Davitt, A., Volpato, G., & Bradbury, K. (2023). High-resolution global building emissions estimation using satellite imagery. Presented at the NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning.</sub>*


### 2.4  Goals  <a name="Goals"></a>

The goal of this project is to develop a machine learning model to predict the Energy Use Intensity (EUI) of buildings using globally available climatic, geographical, and socioeconomic features. These EUI predictions serve as a foundation for estimating global GHG emissions from buildings in future work.

Energy Use Intensity is a key metric representing the amount of energy consumed per unit area of a building. To evaluate prediction accuracy, we use the Mean Absolute Percentage Error (MAPE), where lower values indicate better model performance. Therefore, minimizing MAPE is a central objective of this work.

This project focuses on developing the Energy Use Intensity (EUI) estimation technique using globally available features. By selecting key features, the goal has been to improve EUI predictions and reduce the average MAPE in the cross-validation context.

### 2.5 Project Deliverables and Presentation Materials <a name="ProjectDeliverablesAndPresentationMaterials"></a>

This section provides an overview of the key deliverables and presentation materials developed throughout the project. These materials summarize the project's progress, next steps, and areas to explore in the upcoming semester, offering insights into the work completed and the outcomes achieved.

2024 Fall Semester:

1. The deliverables we have agreed to provide to our client this semester can be found [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/tree/main/deliverables_agreement).

2. For a visual summary of the project, check out the slide deck presentation [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/slide_decks/Climate_TRACE_Presentation.pdf).

3. The mid-point slide deck of analysis and results can be found [here](https://docs.google.com/presentation/d/1aeell_KmJwJAF3aopTo4ghbe1blzv2hjZ7Oo8uiPV3E/edit?usp=sharing).


2025 Spring Semester:

1. The deliverables we have agreed to provide to our client this semester can be found [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/deliverables_agreement/Capstone_Spring_Semester_Plan.pdf).

2. For a visual summary of the project, the final symposium slide deck of analysis and results can be found [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/slide_decks/Final_Climate%20Trace_Presentation.pdf).


## 3. Background <a name="Background"></a>

The accurate estimation of anthropogenic CO₂ emissions is critical for understanding global climate change and formulating effective policies. Existing estimates are provided by several key datasets, including the Open-source Data Inventory for Anthropogenic CO₂ ([Oda, Maksyutov, & Andres, 2018](#oda2018)), the Community Emissions Data System ([McDuffie et al., 2020](#mcduffie2020)), the Emissions Database for Global Atmospheric Research (EDGAR) ([Janssens-Maenhout et al., 2019](#janssens2019)), the Global Carbon Grid ([Tong et al., 2018](#tong2018)), and the Global Gridded Daily CO₂ Emissions Dataset ([Dou et al., 2022](#dou2022)). While the GRACED data is updated nearly monthly, most of the other key datasets suffer from a significant production latency, often of a year or more. Additionally, the highest resolution available across these datasets is 0.1 decimal degrees, roughly equivalent to an 11 km grid near the equator. Furthermore, only a few of these datasets provide detailed breakdowns of emissions by sector, such as residential and commercial subsectors, or offer separate estimates for different greenhouse gases([Markakis et al., 2023](#markakis2023)).

In response to these challenges, recent advancements have been made in the development of more granular and timely emissions estimation methods. One such breakthrough is the High-resolution Global Building Emissions Estimation using Satellite Imagery model by [Markakis et al. (2023)](#markakis2023). This innovative model offers high-resolution, global emissions estimates for both residential and commercial buildings at a 1 km² resolution, with updates on a monthly basis. By leveraging satellite imagery-derived features and machine learning techniques, the model estimates direct emissions from buildings. This approach addresses the temporal and spatial limitations of previous datasets by predicting building areas, estimating energy use intensity, and calculating emissions based on regional fuel mixes. Unlike other datasets like GRACED and EDGAR, this model offers more granular insights into emissions at a higher frequency and resolution, making it a crucial tool for policymakers working to reduce emissions in the building sector on a global scale.

## 4. Methodology <a name="Methodology"></a>
### 4.1 Overall Framework  <a name="OverallFramework"></a>   
Overall, we develop feature map to define the solution search boundaries, employ geographic information techniques, image embedding retrieval methods, and standard statistical techniques to process features, apply supervised learning models to predict target variables, and conduct cross-region evaluations to provide conservative prediction.
![methodology overview](/figures/methodology-1.jpg)
** Grid image from Markakis, P. J., Gowdy, T., Malof, J. M., Collins, L., Davitt, A., Volpato, G., & Bradbury, K. (n.d.). High-resolution Global Building Emissions Estimation using Satellite Imagery.

### 4.2 Feature Map  <a name="FeatureMap"></a>
We develop the feature map by identifying variables that potentially affect the target variable. Starting from previous studies, we conduct a literature review and further interpret the project from both personal and professional perspectives. Considering data coverage and availability, we summarize our features into four main categories: building geometry data, weather, socioeconomics, and policy/law.
![feature map](/figures/methodology-2.jpg)

1. **[Image Embedding](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/data/03_processed/merged_df.csv)**:Embeddings from the pretrained Clay model represent image-based features. We compress these features via PCA (n=1) to extract the principal component score and/or apply KNN to assign cluster memberships and/or Fuzzy C-means to compute the maximum probability score of the assigned cluster for downstream prediction tasks.

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

### 4.3 Feature Engineering  <a name="FeatureEngineering"></a>
Feature engineering is essential to transform raw data into meaningful representations that enhance model performance and predictive accuracy. In this study, we applied the following techniques: 

1. **Geographic information techniques:**
   
    - **Heating Degree Days Calculation:** Retrieve air temperature at 2m above the surface, data source from European Commission and the Group on Earth Observations. Calculate HDD to measure the demand for heating energy based on the difference between outdoor temperature and a baseline "comfort" temperature, typically 65°F (18°C).
    - **Cooling Degree Days Calculation:** Retrieve air temperature at 2m above the surface, data source from European Commission and the Group on Earth Observations. Calculated using temperature data to derive features measure the demand for Cooling related energy usage based on the difference between outdoor temperature and a baseline "comfort" temperature, typically 65°F (18°C).
   
**Note:Temperature, DewPoint Temperature are also developed with similar techniques.

2. **Image Embedding Retreval Method:**
   
   - Retrieve satellite images from Sentinel-2 for the locations of interest.
   - Employ Clay (a pretrained Vision Transformer MAE model specialized in satellite images) to generate embeddings.
   - Apply dimension reduction(PCA) and/or classification methods(Fuzzy Classter;KNN;GMM) to optimize the representational ability of embeddings.
   - These derived features encode spatial, visual, and structural patterns for downstream prediction task.
      
**Note:Clay model related materials(GitHub link for env info, introductory video for clay): https://clay-foundation.github.io/model/index.html

3. **Standard Statistical Techniques:**
   
    - **Nearest Reference Mapping:** Nearest Reference Mapping involves assign each data point to its closest reference location based on a defined distance metric, enriching the dataset with relevant features from these reference points. In this project, we aim to assigning **EUI values** to each data point based on its nearest starting point with known ground truth. By using the EUI values as features and incorporating spatial context into our model, we aim to improve the model’s starting point and enhance prediction accuracy for global projections. 
    - **GDP per Capita Calculation:** We use GDP per capita, which is the result of dividing total GDP by the population, as it provides more relevant information for our model. This approach better captures the economic impact on energy consumption at the individual level, enabling more accurate comparisons across regions with varying population sizes.
    - **Sin/Cos transformation:** To preserve directionality and spatial information, we transform latitude and longitude using sine and cosine functions. Specifically, we convert lat/lon to radians and compute their sine and cosine values as input features.This accounts for the circular nature of geographic coordinates and helps the model capture spatial proximity. However, after experimentation, this approach did not yield better performance compared to using raw latitude and longitude values. Therefore, we reverted to using the standard lat/lon features in our final models.
     
**Note:HDI, GDP, Education, Income, Urbanization, and Paris Agreement features are processed using statistical techniques suited to their characteristics.

**After feature engineering and merging our datasets, the full process is available in the Jupyter notebook [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/notebooks/030_DataPreprocessing.ipynb). This resulted in the final dataset for model input, containing 482 data points, which can be accessed [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/data/03_processed/merged_df.csv)

## 5. Models <a name="Models"></a>   

### 5.1 Supervised Machine Learning <a name="SupervisedMachineLearning"></a>  

In this project, we will employ a range of supervised machine learning models to predict and analyze the target variable. The following models will be utilized:

1. **Linear Regression:**  
   We will use Linear Regression to model the relationship between the input features and the target variable. This model is suitable for capturing linear relationships and will serve as a baseline for comparison with more complex models.

2. **K-Nearest Neighbors (KNN):**  
   KNN is a non-parametric model that classifies a data point based on the majority class or average value of its nearest neighbors. It is particularly useful for capturing local patterns in the data and will provide a comparison to Linear Regression in terms of flexibility.

3. **Ensemble Models:**

   - **Random Forest:** Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. It is particularly useful for handling high-dimensional data and capturing complex, non-linear relationships.
   - **XGBoost:** XGBoost is an optimized gradient boosting algorithm that performs well in a variety of prediction tasks. It builds an ensemble of decision trees sequentially, improving the model’s performance by reducing bias and variance.
   
   - **CatBoost:** CatBoost is another gradient boosting algorithm known for its handling of categorical features without the need for explicit preprocessing. It is expected to provide competitive results, particularly in datasets with mixed types of variables.

The combination of linear models, distance-based methods like KNN, and powerful ensemble models like XGBoost and CatBoost will allow us to capture a range of patterns in the data, from simple linear trends to more complex interactions and non-linear relationships.

### 5.2 Experimental Design <a name="ExperimentalDesign"></a>
1. **Feature Selection**
Removed less important features based on feature importance analysis.

3. **Hyperparameter Tuning**
Grid Search for MAPE optimization.

4. **Validation Strategy**
Given the challenge of regional variations in global data, we will validate our predictions at the regional level across five distinct regions using three strategies to identify biases and improve model robustness. The regions we are using are defined as follows:
Given the challenge of regional variations in global data, we will validate our predictions at the regional level across five distinct regions using three strategies to identify biases and improve model robustness. This approach helps to account for local differences in energy use patterns and improve the model’s predictive accuracy across diverse contexts. The regions we are using are defined as follows:

   - **Asia & Oceania**
   - **Europe**
   - **Africa**
   - **Central and South America**
   - **Northern America**

   The data points in our dataset, which we intend to predict, are distributed across these various regions, as illustrated in the following map.


![Geographic Distribution of Data Points by Region](/figures/03_region_map.png)
   
   The strategies we will be using are as follows:

![Image](/figures/04_experimental_design.png)
   
   We aim to assess our model's generalization by comparing its performance within the same region (Within-Domain) and its ability to extrapolate to other regions (Cross-Domain). The goal is to reduce the gap between these strategies to improve accuracy and understand extrapolation errors. Additionally, we want to understand if there are regions that perform better than others in specific outcomes, which can help us tailor our model to regional differences.

## 6. Experiments <a name="Experiments"></a>

### 6.1 Feature Importance <a name="FeatureImportance"></a>

To identify the most influential factors in building energy use and greenhouse gas emissions, we used a Random Forest model. The input variables included socioeconomic, climatic, and geographic indicators. Among the evaluated features, the Income Index (32.09%) and Average Temperature (23.58%) emerged as the most important predictors of energy use intensity (EUI), followed by Latitude (16.41%) and Average Dewpoint Temperature (6.53%). These results highlight the strong influence of income levels and climate on building energy consumption. In contrast, features such as GDP per capita (0.45%), the Paris Agreement indicator (0.01%), and image-based embeddings contributed very little, suggesting that image-derived features were not particularly informative in this context.

We initially tested our models using all available features and then evaluated performance by selecting the most important ones. After testing several options, we decided to set a threshold to retain only the features contributing more than 1% to the model's predictions. This feature selection process helped streamline the model, focusing on the most influential variables while improving computational efficiency.

For details on the calculations, check the [Feature Importance Notebook](/notebooks/050_FeatureImportance.ipynb).   

![Feature Importance](/figures/05_feature_importance.png)

### 6.2 Models <a name="Models"></a>

In this section, we evaluate the performance of several machine learning models used for predicting Energy Use Intensity (EUI) and estimating greenhouse gas (GHG) emissions from buildings. The models tested include Linear Regression (LR), Linear Regression with Lasso and Ridge regularization, K-Nearest Neighbors (KNN), Random Forest, XGBoost, and CatBoost. The evaluation metrics, such as Mean Absolute Percentage Error (MAPE) and R², are used to assess model performance across different feature sets. The models are also evaluated across various cross-validation strategies to ensure robustness and generalizability. 

In particular, we select the best model based on **minimizing MAPE in the cross domain strategy**, as it better reflects the model’s real-world generalization ability and allows us to optimize accordingly.

As a **baseline**, we consider the scenario where EUI is predicted simply by using the value from the geographically closest data point. To implement this, we use a **K-Nearest Neighbors model with K=1, using only latitude and longitude** as input features. With this approach, we obtained an **average MAPE of 37.8%**, which serves as our reference point for evaluating model improvements.

We initially tested our models using all available features and then evaluated performance by selecting only the most important ones. After testing several options, we decided to set a threshold to retain only features that contributed more than 1% to the model's predictions. The summary results of both alternatives—using all features versus using only those with more than 1% importance—are presented below.


#### **Model Performance Using All Features**
The specific features utilized in each model, along with the hyperparameters tested, can be found in detail in the tables [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/tree/main/results/all_features) and are summarized in this [table](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/results/all_features/comparison_average_results.csv).

The following graphs display the average performance metrics for MAPE, R², and RMSE for both residential and non-residential buildings across different cross-validation strategies: within domain, cross-domain, and all domain. These averages are calculated for various regions, helping us to identify the best model and select our EUI Estimation Technique.

   - **MAPE**  

![eui_predictions_all_domain](/figures/model_plots/all_features/00_model_comparison_mape.png)

   - **R²**  

![eui_predictions_all_domain](/figures/model_plots/all_features/00_model_comparison_r2.png)

   - **RMSE** 

![eui_predictions_all_domain](/figures/model_plots/all_features/00_model_comparison_rmse.png)


#### **Model Performance After Feature Selection**
The specific features utilized in each model, along with the hyperparameters tested, can be found in detail in the tables [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/tree/main/results/selected_features) and are summarized in this [table](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/results/selected_features/comparison_average_results.csv).

The following graphs display the average performance metrics for MAPE, R², and RMSE for both residential and non-residential buildings across different cross-validation strategies: within domain, cross-domain, and all domain. These averages are calculated for various regions, helping us to identify the best model and select our EUI Estimation Technique.

   - **MAPE**  
![eui_predictions_all_domain](/figures/model_plots/selected_features/00_model_comparison_mape.png)

   - **R²**  
![eui_predictions_all_domain](/figures/model_plots/selected_features/00_model_comparison_r2.png)

   - **RMSE** 
![eui_predictions_all_domain](/figures/model_plots/selected_features/00_model_comparison_rmse.png)



#### Best Model Overall: Random Forest - With Feature Selection

The following table summarizes the MAPE results under the cross-domain validation strategy for residential and non-residential buildings, highlighting the average performance across both.

| Model                           | Non-residential MAPE (Cross-domain) | Residential MAPE (Cross-domain) | Average MAPE  (Cross-domain) |
|--------------------------------|--------------------------------------|----------------------------------|--------------|
| KNN (K=1, Lat & Long)     | 38.4%                               | 37.2%                           | 37.8%        |
| CAT Boost (All Features)       | 17.2%                               | 20.7%                           | 18.95%       |
| Random Forest (Top Features)   | 13.8%                               | 21.2%                           | 17.51%       |




Based on our evaluation across metrics, we selected Random Forest as our primary model for EUI prediction. While some models occasionally outperformed in specific scenarios, Random Forest demonstrated the most consistent and balanced performance across validation strategies and building types. It achieved MAPE values below 15% for non-residential and 21% for residential in within-domain validation, maintained positive R² values (0.22-0.52 within-domain), and showed stable RMSE values (29.3 non-residential, 23.6 residential within-domain). This consistent performance, along with its ability to handle non-linear relationships and maintain stability in cross-domain scenarios, makes Random Forest the most reliable choice for global EUI prediction.


The following figure shows detailed performance metrics for the Random Forest model across different validation strategies and building types. Detailed results by region, along with the estimation technique used, including the specific variables and their hyperparameters, can be found in this [table](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/results/20241208_2046_rf_detailed_results.csv), while average performance metrics are available [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/results/20241208_2046_rf_average_results.csv).

![eui_predictions_all_domain](/figures/06_avg_rf.png)

A detailed analysis of the Random Forest model's performance revealed distinct patterns across building types and validation strategies. For non-residential buildings, the model achieved its best performance in within-domain validation with a MAPE of 8.96% and R² of 0.22, though performance declined in cross-domain scenarios (MAPE 13.58%, R² -0.36). For residential buildings, while showing higher error rates (MAPE 12.76% within-domain), it demonstrated stronger explanatory power (R² 0.52). The all-domain strategy provided a balanced middle ground, with MAPE of 9.98% and 13.08% for non-residential and residential buildings respectively. These results demonstrate that while geographical variations impact model performance, the Random Forest consistently maintains error levels well within our target range of 30-40% MAPE across all scenarios, making it a robust choice for global EUI prediction.

To better understand the Random Forest model's performance across different validation strategies, we examine the relationship between predicted and actual EUI values, along with error distributions for each region. The following figures show these relationships for within-domain, cross-domain, and all-domain validation approaches. For each strategy, we present both scatter plots comparing predicted versus actual values, and corresponding error distribution histograms, broken down by geographical region and building type.

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


The regional analysis shows that Asia & Oceania consistently demonstrates one of the best overall performances across validation strategies, with low MAPE (18.2% residential, 6.4% non-residential) and strong R² values (0.69 residential, 0.85 non-residential) in within-domain validation. This performance remains relatively stable in cross-domain validation (MAPE 35.3% residential, 13.7% non-residential; R² 0.44 residential, 0.52 non-residential), outperforming other regions. 

Central and South America also stands out with strong performance in both within-domain (MAPE 4.2% residential, 3.2% non-residential; R² 0.85 residential, 0.46 non-residential) and cross-domain validation (MAPE 10.6% residential, 5.4% non-residential; R² -0.05 residential, -0.47 non-residential), though with more variable R² values. 

Other regions show more variable performance, with Europe, Northern America and Africa having higher error rates and less consistent R² values (Africa showing MAPE of 7.6-8.2% but poor R² values near zero in most validation strategies). 


## 7. Conclusion <a name="Conclusion"></a>

Our analysis reveals significant insights into developing machine learning approaches for global EUI prediction. Ensemble models, particularly Random Forest, consistently outperformed traditional methods across validation strategies, achieving MAPE values below 20% and surpassing our initial target of 30-40%. However, the variation in R² values, especially in cross-domain scenarios, indicates challenges in capturing the full complexity of EUI patterns across different regions.

Regional analysis uncovered important patterns in model performance. Asia & Oceania and Central/South America demonstrated the strongest results, while Europe and Northern America showed more variable predictions. Africa presented an interesting case with low error rates but poor explanatory power. The significant performance differences between within-domain and cross-domain validation highlight the strong influence of regional characteristics on EUI predictions.

The technical insights gained suggest strongly non-linear relationships between features and EUI, reinforcing the necessity of sophisticated modeling approaches. After evaluating several algorithms, we found
that the Random Forest model delivered the best performance. Among the features, Income Index, Average Temperature, and Latitude emerged as the most influential in predicting energy use intensity. These results can support future researchers in generating high-resolution EUI predictions to improve emissions estimates while also helping policymakers better understand spatial patterns of energy consumption and design more targeted emission reduction strategies.   


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
│   │   ├── CDD_scalematched.csv
│   │   ├── HDD_scalematched.csv
│   │   ├── eui_boston.csv
│   │   ├── eui_california.csv
│   │   ├── eui_chicago.csv
│   │   ├── eui_miami.csv
│   │   ├── eui_nyc.csv
│   │   ├── eui_philadephia.csv
│   │   ├── eui_seattle.csv
│   │   ├── eui_usa_cities_grouped_df.csv
│   │   ├── humidity_backup.csv
│   │   ├── image_results.csv
│   │   ├── population_density.csv
│   │   └── temperature_dewpoint_precipitation_2023.csv
│   └── 03_processed
│   │   ├── merged_df.csv
│   │   ├── train_test_split_new_data.csv
│       └── train_test_split_original_data.csv
├── deliverables_agreement
│   │   ├── Capstone_Spring_Semester_Plan.pdf
│       └── Mid-Point Deliverables - Climate Trace.pdf
├── figures
│   ├── model_plots
│   │   ├── 00_model_comparison_mape.png
│   │   ├── 00_model_comparison_r2.png
│   │   ├── 00_model_comparison_rmse.png
│   │   ├── cat_all_domain_error_distribution.png
│   │   ├── cat_all_domain_eui_predictions.png
│   │   ├── cat_cross_domain_error_distribution.png
│   │   ├── cat_cross_domain_eui_predictions.png
│   │   ├── cat_within_domain_error_distribution.png
│   │   ├── cat_within_domain_eui_predictions.png
│   │   ├── knn_all_domain_error_distribution.png
│   │   ├── knn_all_domain_eui_predictions.png
│   │   ├── knn_cross_domain_error_distribution.png
│   │   ├── knn_cross_domain_eui_predictions.png
│   │   ├── knn_within_domain_error_distribution.png
│   │   ├── knn_within_domain_eui_predictions.png
│   │   ├── lasso_all_domain_error_distribution.png
│   │   ├── lasso_all_domain_eui_predictions.png
│   │   ├── lasso_cross_domain_error_distribution.png
│   │   ├── lasso_cross_domain_eui_predictions.png
│   │   ├── lasso_within_domain_error_distribution.png
│   │   ├── lasso_within_domain_eui_predictions.png
│   │   ├── lr_all_domain_error_distribution.png
│   │   ├── lr_all_domain_eui_predictions.png
│   │   ├── lr_cross_domain_error_distribution.png
│   │   ├── lr_cross_domain_eui_predictions.png
│   │   ├── lr_within_domain_error_distribution.png
│   │   ├── lr_within_domain_eui_predictions.png
│   │   ├── rf_all_domain_error_distribution.png
│   │   ├── rf_all_domain_eui_predictions.png
│   │   ├── rf_cross_domain_error_distribution.png
│   │   ├── rf_cross_domain_eui_predictions.png
│   │   ├── rf_within_domain_error_distribution.png
│   │   ├── rf_within_domain_eui_predictions.png
│   │   ├── ridge_all_domain_error_distribution.png
│   │   ├── ridge_all_domain_eui_predictions.png
│   │   ├── ridge_cross_domain_error_distribution.png
│   │   ├── ridge_cross_domain_eui_predictions.png
│   │   ├── ridge_within_domain_error_distribution.png
│   │   ├── ridge_within_domain_eui_predictions.png
│   │   ├── xgb_all_domain_error_distribution.png
│   │   ├── xgb_all_domain_eui_predictions.png
│   │   ├── xgb_cross_domain_error_distribution.png
│   │   ├── xgb_cross_domain_eui_predictions.png
│   │   ├── xgb_within_domain_error_distribution.png
│   │   └── xgb_within_domain_eui_predictions.png
│   ├── 01_formula.png
│   ├── 02_eui_map.png
│   ├── 03_region_map.png
│   ├── 04_experimental_design.png
│   ├── 05_feature_importance.png
│   ├── 06_avg_rf.png
│   ├── methodology-1.jpg
│   ├── methodology-2.jpg
│   ├── methodology-3.jpg
│   ├── methodology-4.jpg
│   └── methodology-5.jpg
├── notebooks
│   ├── catboost_info
│   │   ├── learn
│   │   │   └── events.out.tfevents
│   │   ├── catboost_training.json
│   │   ├── learn_error.tsv
│   │   └── time_left.tsv
│   ├── 010_Download_WeatherData_API.ipynb
│   ├── 011_EUIBostonProcessing.ipynb
│   ├── 012_EUISeattleProcessing.ipynb
│   ├── 014_EUICaliforniaProcessing.ipynb
│   ├── 015_EUINYCProcessing.ipynb
│   ├── 016_EUIChicagoProcessing.ipynb
│   ├── 017_EUIPhiladelphiaProcessing.ipynb
│   ├── 018_EUIMiamiProcessing.ipynb
│   ├── 019_MergeEUI.ipynb
│   ├── 020_WeatherData_Preprocessing.ipynb
│   ├── 021_HumidityPreprocessing.ipynb
│   ├── 023_HDDPreprocessing.ipynb
│   ├── 024_CDDPreprocessing.ipynb
│   ├── 025_Population.ipynb
│   ├── 030_DataPreprocessing.ipynb
│   ├── 040_Plots.ipynb
│   ├── 050_FeatureImportance.ipynb
│   ├── 060_Experiments_LR.ipynb
│   ├── 061_Experiments_KNN.ipynb
│   ├── 062_Experiments_RF.ipynb
│   ├── 062_Experiments_RF_GridSearch.ipynb
│   ├── 063_Experiments_XGBoost.ipynb
│   ├── 063_Experiments_XGBoost_GridSearch.ipynb
│   ├── 064_Experiments_CatBoost.ipynb
│   ├── 064_Experiments_CatBoost_GridSearch.ipynb
│   ├── 070_Model_Comparison.ipynb
│   └── iamge-embeddingv2.ipynb
├── requirements.txt
├── results
│   ├── gridsearch
│   │   ├── 20250326_1249_rf_grid_search_results.csv
│   │   └── 20250326_1522_cat_grid_search_results.csv
│   ├── .result.txt
│   ├── 20250325_2018_lasso_average_results.csv
│   ├── 20250325_2018_lasso_detailed_results.csv
│   ├── 20250325_2018_lr_average_results.csv
│   ├── 20250325_2018_lr_detailed_results.csv
│   ├── 20250325_2018_ridge_average_results.csv
│   ├── 20250325_2018_ridge_detailed_results.csv
│   ├── 20250325_2047_knn_average_results.csv
│   ├── 20250325_2047_knn_detailed_results.csv
│   ├── 20250325_2109_cat_average_results.csv
│   ├── 20250325_2109_cat_detailed_results.csv
│   ├── 20250326_1551_xgb_average_results.csv
│   ├── 20250326_1551_xgb_detailed_results.csv
│   ├── 20250403_1440_rf_average_results.csv
│   ├── 20250403_1440_rf_detailed_results.csv
│   └── comparison_average_results.csv
├── slide_decks
│   ├── Climate_TRACE_Presentation.pdf
│   └── Final_Climate Trace_Presentation.pdf
└── src
    ├── __init__.py
    ├── __pycache__
    │   └── lib.cpython-311.pyc
    └── lib.py

```


1. **`data/`**  
   - Contains all datasets used in the project. It is organized into subfolders:
     - **01_raw/**: Raw, unprocessed datasets like HDI, GDP, and population data.  
     - **02_interim/**: Intermediate files including HDD, CDD, humidity, and city-level EUI datasets.  
     - **03_processed/**: Fully processed datasets (e.g., merged_df.csv) ready for modeling and evaluation.  

2. **`figures/`**  
     - Includes all visual materials such as model evaluation plots, maps, diagrams, and methodology figures.
       - **model_plots/**: Contains prediction visualizations and error distributions for all tested models across within, cross, and all-domain scenarios.
         
3. **`notebooks/`**  

   - Jupyter notebooks used for data processing, feature engineering, modeling, and analysis. Notebooks are ordered and labeled for clarity:  
     - **010_Download_WeatherData_API.ipynb**: Downloads weather data from the Copernicus Climate Data Store (ERA5-Land daily statistics).
     - **011–019**: Download and process regional EUI datasets (e.g., Boston, Seattle, NYC).
     - **020–025**: Process weather, humidity, HDD, CDD, and population data for modeling. 
     - **030_DataPreprocessing.ipynb**: Prepares the final dataset for model input.  
     - **040_Plots.ipynb**: Generates visualizations for feature trends and region-level insights as well as for analysis and reporting.  
     - **050_FeatureImportance.ipynb**: Analyzes feature importance for model evaluation.  
     - **060–070**: Run experiments with Linear Regression, KNN, Random Forest, XGBoost, CatBoost, and compare model performance. 

4. **`results/`**  
     - Stores results from all model runs including:
       - Average and detailed results from each model and domain. 
       - Grid search output from hyperparameter tuning.  
       - Final model comparison tables (`comparison_average_results.csv`).

5. **`deliverables_agreement/`**  
   - Documents submitted to course instructors for planning and mid-point check-ins.   


6. **`slide_decks/`**  
   - Final and mid-point presentation slides in PDF format.   
     
7. **`src/`**  
   - Contains core Python scripts for the project.  
     - **lib.py**: Provides utility functions and shared modules for data preprocessing, feature extraction, and model evaluation, used across notebooks and scripts.  

8. **`requirements.txt`**  
   - Lists all dependencies needed for the project environment, ensuring reproducibility.  

9. **`README.md`**  
   - The entry point of the repository, providing an overview, key results, and links to all major components.  


### Usage Instructions  

1. **Setup**:  
   Clone the repository and ensure all dependencies are installed. Use `requirements.txt`
      ```bash
   git clone git@github.com:AliciaXia222/Capstone-Team-Climate-Trace.git
   cd Capstone-Team-Climate-Trace
   pip install -r requirements.txt   

2. **Data Processing**:  
   - Start with `010_Download_WeatherData_API.ipynb` to download raw weather data from the Copernicus Climate Data Store.  
   - Use `020_WeatherData_Preprocessing.ipynb` to preprocess the weather data for model input.  
   - Process specific features with `021_HumidityPreprocessing.ipynb`, `023_HDDPreprocessing.ipynb`, and `024_CDDPreprocessing.ipynb` to compute humidity, Heating Degree Days (HDD), and Cooling Degree Days (CDD) data.  
   - Finalize the dataset with `030_DataPreprocessing.ipynb` before moving to modeling.

4. **Modeling**:  
   - Open `060_Experiments_LR.ipynb` and `061_Experiments_KNN.ipynb`, etc., to train individual models and evaluate performance across domains.
   - `070_Model_Comparison.ipynb` to compare models across domains. 

5. **Results Analysis**:  
   - Use the `results/` directory to analyze model outputs and metrics.
   - Use figures in figures/model_plots/ for performance visualizations.   
 

6. **Figures and Visuals**:  
   - All generated plots and diagrams are stored in `figures/` for easy reference in presentations or reports.  


## 9. Resources  <a name="Resources"></a>

1. [Climate TRACE. (2023). *More than 70,000 of the highest-emitting greenhouse gas sources in the world are now tracked by Climate TRACE*.](https://climatetrace.org/news/more-than-70000-of-the-highest-emitting-greenhouse-gas) <a name="climatetrace2023"></a>

2. [Dou, X., Wang, Y., Ciais, P., Chevallier, F., Davis, S. J., Crippa, M., Janssens-Maenhout, G., Guizzardi, D., Solazzo, E., Yan, F., Huo, D., Zheng, B., Zhu, B., Cui, D., Ke, P., Sun, T., Wang, H., Zhang, Q., Gentine, P., Deng, Z., & Liu, Z. (2022). Near-realtime global gridded daily CO2 emissions. *The Innovation, 3*(1), 100182.](https://doi.org/10.1016/j.xinn.2021.100182) <a name="dou2022"></a>

3. [Janssens-Maenhout, G., Crippa, M., Guizzardi, D., Muntean, M., Schaaf, E., Dentener, F., Bergamaschi, P., Pagliari, V., Olivier, J. G. J., Peters, J. A. H. W., van Aardenne, J. A., Monni, S., Doering, U., Petrescu, A. M. R., Solazzo, E., & Oreggioni, G. D. (2019). EDGAR v4.3.2 Global Atlas of the three major greenhouse gas emissions for the period 1970–2012. *Earth System Science Data, 11*(3), 959–1002.](https://doi.org/10.5194/essd-11-959-2019) <a name="janssens2019"></a>

4. [Live Science. (2024, March 5). *Global carbon emissions reach new record high in 2024*.](https://www.livescience.com/planet-earth/climate-change/global-carbon-emissions-reach-new-record-high-in-2024-with-no-end-in-sight-scientists-say) <a name="livescience2024"></a>

5. [Luers, A., Yona, L., Field, C. B., Jackson, R. B., Mach, K. J., Cashore, B. W., Elliott, C., Gifford, L., Honigsberg, C., Klaassen, L., & Matthews, H. D. (2022). *Make greenhouse-gas accounting reliable—Build interoperable systems*. *Nature, 607*(7920), 653–656.](https://pubmed.ncbi.nlm.nih.gov/35882990/) <a name="luers2022"></a>

6. [Markakis, P. J., Gowdy, T., Malof, J. M., Collins, L., Davitt, A., Volpato, G., & Bradbury, K. (2023). High-resolution global building emissions estimation using satellite imagery. *Climate Change AI*.](https://www.climatechange.ai/papers/neurips2023/128/paper.pdf) <a name="markakis2023"></a>

7. [McDuffie, E. E., Smith, S. J., O’Rourke, P., Tibrewal, K., Venkataraman, C., Marais, E. A., Zheng, B., Crippa, M., Brauer, M., & Martin, R. V. (2020). A global anthropogenic emission inventory of atmospheric pollutants from sector- and fuel-specific sources (1970–2017): An application of the Community Emissions Data System (CEDS). *Earth System Science Data, 12*(4), 3413–3442.](https://doi.org/10.5194/essd-12-3413-2020) <a name="mcduffie2020"></a>

8. [Oda, T., Maksyutov, S., & Andres, R. J. (2018). The Open-source Data Inventory for Anthropogenic CO2, version 2016 (ODIAC2016): A global monthly fossil fuel CO2 gridded emissions data product for tracer transport simulations and surface flux inversions. *Earth System Science Data, 10*(1), 87–107.](https://doi.org/10.5194/essd-10-87-2018) <a name="oda2018"></a>

9. [Tong, D., Zhang, Q., Davis, S. J., Liu, F., Zheng, B., Geng, G., Xue, T., Li, M., Hong, C., Lu, Z., Streets, D. G., Guan, D., & He, K. (2018). Targeted emission reductions from global super-polluting power plant units. *Nature Sustainability, 1*(1), 59–68.](https://doi.org/10.1038/s41893-017-0003-y) <a name="tong2018"></a>


## 10. Contributors  <a name="Contributors"></a>
[Jiechen Li](https://github.com/carrieli15)  
[Meixiang Du](https://github.com/dumeixiang)  
[Yulei Xia](https://github.com/AliciaXia222)  
[Barbara Flores](https://github.com/BarbaraPFloresRios)  


#### Project Mentor and Client: [Dr. Kyle Bradbury](https://energy.duke.edu/about/staff/kyle-bradbury)
