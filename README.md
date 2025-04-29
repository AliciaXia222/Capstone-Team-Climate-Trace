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
5. [Machine Learning Approach and Experimental Setup](#MachineLearningApproachandExperimentalSetup)  
   5.1 [Supervised Machine Learning](#SupervisedMachineLearning)    
   5.2 [Experimental Design](#ExperimentalDesign)
6. [Experiments](#Experiments)  
   6.1 [Number of Clusters in Fuzzy C-Means](#NumberofClustersinFuzzyCMeans)  
   6.2 [Feature Importance](#FeatureImportance)  
   6.3 [Models](#Models)    
7. [Conclusion and Discussion](#Conclusion)
8. [Repository Structure and Usage](#RepositoryStructureAndUsage)
9. [Resources](#Resources)
10. [Contributors](#Contributors)   

## 1. Abstract <a name="Abstract"></a>

This project develops a machine learning model to estimate direct greenhouse gas (GHG) emissions from residential and non-residential building energy consumption. The model predicts energy use intensity (EUI) by incorporating climatic, geographical, and socioeconomic variables. These EUI estimates, combined with global building floor area, serve as a basis for calculating direct GHG emissions from buildings, offering a timely, high-resolution approach to global emissions estimation.

The work focuses on developing an EUI estimation technique, with an emphasis on minimizing the Mean Absolute Percentage Error (MAPE). Starting from a baseline K-Nearest Neighbors (KNN) model (K = 1), using only geographic location (latitude and longitude), with an average MAPE of 37.8% in cross-domain validation, we reduced the error by training a Random Forest model. To further improve performance, we applied grid search to optimize hyperparameters and iterated over different combinations of image embedding clusterings. As a result, we achieved an average MAPE of 17.9%. This represents a 53% improvement from the baseline in cross-domain validation, which is the most conservative strategy compared to all-domain and within-domain evaluations. These results highlight the robustness of the model and the effectiveness of the proposed methodology.

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

1. **[Image Embedding](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/notebooks/026_ImageEmbedding.ipynb)**:Embeddings from the pretrained Clay model represent image-based features. We compress these features via PCA (n=1) to extract the principal component score and/or apply KNN to assign cluster memberships and/or Fuzzy C-means to compute the maximum probability score of the assigned cluster for downstream prediction tasks.

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
![imageembedding](/figures/satteliteimage.jpg)
   
**Note:[Clay model related materials(GitHub link for env info, introductory video for clay)](https://clay-foundation.github.io/model/index.html)

3. **Standard Statistical Techniques:**
   
    - **Nearest Reference Mapping:** Nearest Reference Mapping involves assign each data point to its closest reference location based on a defined distance metric, enriching the dataset with relevant features from these reference points. In this project, we aim to assigning **EUI values** to each data point based on its nearest starting point with known ground truth. By using the EUI values as features and incorporating spatial context into our model, we aim to improve the model’s starting point and enhance prediction accuracy for global projections. 
    - **GDP per Capita Calculation:** We use GDP per capita, which is the result of dividing total GDP by the population, as it provides more relevant information for our model. This approach better captures the economic impact on energy consumption at the individual level, enabling more accurate comparisons across regions with varying population sizes.
    - **Sin/Cos transformation:** To preserve directionality and spatial information, we transform latitude and longitude using sine and cosine functions. Specifically, we convert lat/lon to radians and compute their sine and cosine values as input features.This accounts for the circular nature of geographic coordinates and helps the model capture spatial proximity. However, after experimentation, this approach did not yield better performance compared to using raw latitude and longitude values. Therefore, we reverted to using the standard lat/lon features in our final models.
     
**Note:HDI, GDP, Education, Income, Urbanization, and Paris Agreement features are processed using statistical techniques suited to their characteristics.

**After feature engineering and merging our datasets, the full process is available in the Jupyter notebook [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/notebooks/030_DataPreprocessing.ipynb). This resulted in the final dataset for model input, containing 482 data points, which can be accessed [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/data/03_processed/merged_df.csv)

## 5. Machine Learning Approach and Experimental Setup <a name="MachineLearningApproachandExperimentalSetup"></a>   

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
1. **Image Cluster Selection**
We tested different numbers of clusters (5, 10, and 20) for the Fuzzy C-Means algorithm to assess the effect of clustering on downstream model performance. While the Fuzzy Partition Coefficient (FPC) provided some insights, the final choice of clusters was guided by empirical performance, where we compared models to identify the configuration that delivered the best results.

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

### 6.1 Number of Clusters in Fuzzy C-Means <a name="NumberofClustersinFuzzyCMeans"></a>
To determine the optimal number of clusters for the fuzzy C-means algorithm, we used the Fuzzy Partition Coefficient (FPC). The FPC measures how well the data points fit into the clusters, with higher values indicating that the data is more clearly grouped. In general, a high FPC means the clusters are more distinct and less overlapping. By examining how the FPC changes as the number of clusters increases, we can assess the overall quality of the clustering.

![FFPC](/figures/05_fuzzy_partition_coefficient.png)

From the graph, we observe that as the number of clusters increases from 2 to 20, the FPC decreases steadily, following a trend close to 
1/n, where n is the number of clusters. This behavior suggests that the data does not naturally separate into distinct clusters, as a clear clustering structure would typically cause the FPC to remain relatively high for some specific number of clusters before dropping. The FPC starting around 0.50 for 2 clusters and gradually decreasing to around 0.05 for 20 clusters indicates that adding more clusters does not significantly improve the separation of the data.

Overall, the trend suggests that the dataset may not exhibit strong inherent clustering, and that the fuzzy memberships are relatively uniform across clusters. However, we proceed to test 5, 10, and 20 clusters to explore whether varying the number of clusters affects the downstream performance of the model.

### 6.2 Feature Importance <a name="FeatureImportance"></a>

To identify the most influential factors in building energy use and greenhouse gas emissions, we used a Random Forest model. In the following image, we compare three different cluster configurations: n=5, n=10, and n=20 clusters. In all three cases, we observe that the most important features remain consistent, although their relative contributions vary depending on the number of clusters used. The **Income Index** is consistently the most influential feature, followed by **Average Temperature** and **Latitude** in all three configurations.

While increasing the number of clusters provides more granular information, the contribution of image-derived features remains relatively small, with socio-economic and climatic factors continuing to dominate. Given the limited size of our dataset, with just over 400 rows, it may be more effective to evaluate the model's performance using fewer clusters to avoid overfitting and improve generalization. However, we will also test configurations with 10 and 20 clusters to assess whether finer clustering offers any additional value

For details on the calculations, check the [Feature Importance Notebook](/notebooks/050_FeatureImportance.ipynb).   

![Feature Importance](/figures/06_feature_importance.png)

### 6.3 Models <a name="Models"></a>

In this section, we evaluate the performance of several machine learning models used for predicting Energy Use Intensity (EUI). The models tested include Linear Regression (LR), Linear Regression with Lasso and Ridge regularization, K-Nearest Neighbors (KNN), Random Forest, XGBoost, and CatBoost. The evaluation metrics, such as Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE) are used to assess model performance across different feature sets. The models are also evaluated across various validation strategies to ensure robustness and generalizability. 

In particular, we select the best model based on **minimizing MAPE in the cross domain strategy**, as it better reflects the model’s real-world generalization ability.

In our cross-domain validation experiment, we observed **R²** values close to 0 or even negative. This can be understood because the model is trained on data from four regions (e.g., regions 1-4) and tested on a completely different region (e.g., region 5). Since the model is using information exclusively from other regions that may not fully represent the characteristics of the target region, it often struggles to generalize, leading to lower-than-expected R² values. In many cases, simply predicting the mean of the test region’s data can perform better than the model, indicating that regional differences play a significant role. This suggests that R² may not be the best metric for evaluating performance in cross-domain settings. While R² is useful for assessing model fit on a single dataset, it doesn’t fully capture the challenge of extrapolating to new regions. A low or negative R² here reflects the model’s difficulty adapting to these regional variations rather than a fundamental failure of the model.

As a **baseline**, we consider the scenario where EUI is predicted simply by using the value from the geographically closest data point. To implement this, we use a **K-Nearest Neighbors model with K=1, using only latitude and longitude** as input features. With this approach, we obtained an **average MAPE of 37.8%**, which serves as our reference point for evaluating model improvements.

We evaluate the impact of varying the number of clusters in the fuzzy C-means algorithm on the downstream model performance. Specifically, we test configurations with 5, 10, and 20 clusters to assess whether finer clustering improves model accuracy without leading to overfitting. The summary results for each clustering setting are presented below.


### **MAPE Comparison Across Cluster Configurations**

The plots below show the Mean Absolute Percentage Error (MAPE) for models trained using features derived from **image embeddings clustered with the fuzzy C-means algorithm**. We tested three configurations: 5, 10, and 20 clusters.

   - **5 Clusters**  
![MAPE - 5 Clusters](/figures/model_plots/5_clusters/00_model_comparison_mape.png)

   - **10 Clusters**  
![MAPE - 10 Clusters](/figures/model_plots/10_clusters/00_model_comparison_mape.png)

   - **20 Clusters**  
![MAPE - 20 Clusters](/figures/model_plots/20_clusters/00_model_comparison_mape.png)

As expected, the within-domain validation shows consistently better performance (lower MAPE values) compared to the cross-domain validation across all models, since models tend to perform better when tested on data similar to their training set. However, the most critical evaluation metric is the cross-domain validation, which tests the model's ability to generalize to unseen geographical regions — providing a conservative but realistic assessment of real-world performance. The all-domain validation serves as an intermediate scenario

In the cross-domain evaluation, all models utilizing the full feature set significantly outperform the baseline KNN model, which uses only latitude and longitude. Tree-based models, particularly Random Forest (RF) and XGBoost (XGB), demonstrate superior performance. With 5 clusters, the RF model achieves a 15.6% MAPE for non-residential buildings and 20.2% for residential buildings, resulting in an average MAPE of 17.9%. Interestingly, the 10-cluster configuration yields nearly identical overall performance (17.9% average MAPE), but with slightly better non-residential predictions (14.5%) and slightly worse residential predictions (21.3%). The 20-cluster configuration shows a performance degradation (20.0% average MAPE), likely indicating overfitting due to excessive feature granularity relative to the dataset size. While XGBoost models demonstrate competitive performance, particularly for the 20-cluster configuration (18.6% average MAPE), the Random Forest models with 5 or 10 clusters provide the best balance of accuracy and generalizability across both building types.

### **RMSE Comparison Across Cluster Configurations**

These RMSE plots correspond to the same image-based clustering strategy, with models trained using 5, 10, and 20 clusters from the image embeddings.

   - **5 Clusters** 
![RMSE - 5 Clusters](/figures/model_plots/5_clusters/00_model_comparison_rmse.png)

   - **10 Clusters** 
![RMSE - 10 Clusters](/figures/model_plots/10_clusters/00_model_comparison_rmse.png)

   - **20 Clusters** 
![RMSE - 20 Clusters](/figures/model_plots/20_clusters/00_model_comparison_rmse.png)



### Best Model Overall: Random Forest with 10 Clusters

The following table summarizes the MAPE results under the cross-domain validation strategy for residential and non-residential buildings, highlighting the average performance across both.

| Model | Details                             | Non-residential MAPE <br> Cross-domain | Residential MAPE <br> Cross-domain | Average MAPE <br> Cross-domain |
|:------|:------------------------------------|:---------------------------------------|:----------------------------------|:-------------------------------|
| KNN   | Baseline (Lat & Long only)          | 38.4%                                  | 37.2%                              | 37.8%                          |
| **RF** | All features + 5 image clusters     | **15.6%**                              | **20.2%**                          | **17.9%**                      |
| **RF** | All features + 10 image clusters    | **14.5%**                              | **21.3%**                          | **17.9%**                      |
| RF    | All features + 20 image clusters    | 16.8%                                  | 23.1%                              | 20.0%                          |
| XGB   | All features + 5 image clusters     | 15.4%                                  | 23.1%                              | 19.3%                          |
| XGB   | All features + 10 image clusters    | 14.4%                                  | 24.0%                              | 19.2%                          |
| XGB   | All features + 20 image clusters    | 14.1%                                  | 23.1%                              | 18.6%                          |


Based on our evaluation across metrics, **we selected Random Forest using top features as our primary model for EUI prediction**. While some models occasionally outperformed in specific scenarios, Random Forest demonstrated the most consistent and balanced performance across validation strategies and building types, particularly in the cross-domain setting, which we aim to minimize. It achieved a MAPE of 13.8% for non-residential and 21.2% for residential buildings under cross-domain validation. This consistent performance, along with its ability to handle non-linear relationships and maintain stability across regions, makes Random Forest the most reliable choice for global EUI prediction.

The following figure shows detailed performance metrics for the Random Forest model across different validation strategies and building types. Detailed results by region, along with the estimation technique used, including the specific variables and their hyperparameters, can be found in this [table](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/results/10_clusters/20250428_1719_rf_detailed_results.csv), while average performance metrics are available [here](https://github.com/AliciaXia222/Capstone-Team-Climate-Trace/blob/main/results/10_clusters/20250428_1719_rf_average_results.csv).

![eui_predictions_all_domain](/figures/07_avg_rf.png)

To better understand the Random Forest model's performance across different validation strategies, we examine the relationship between predicted and actual EUI values, along with error distributions for each region. The following figures show these relationships for within-domain, cross-domain, and all-domain validation approaches. For each strategy, we present both scatter plots comparing predicted versus actual values, and corresponding error distribution histograms, broken down by geographical region and building type.

1. **Within Domain**:  

   - **Actual EUI vs. Predicted EUI**  
![eui_predictions_all_domain](/figures/model_plots/10_clusters/rf_within_domain_eui_predictions.png)

   - **Error Distribution Plot**  
![eui_predictions_all_domain](/figures/model_plots/10_clusters/rf_within_domain_error_distribution.png)

2. **Cross Domain**:  

   - **Actual EUI vs. Predicted EUI** 
![eui_predictions_cross_domain](/figures/model_plots/10_clusters/rf_cross_domain_eui_predictions.png)

   - **Error Distribution Plot**  
![eui_predictions_all_domain](/figures/model_plots/10_clusters/rf_cross_domain_error_distribution.png)


3. **All Domain**:  

   - **Actual EUI vs. Predicted EUI** 
![eui_predictions_all_domain](/figures/model_plots/10_clusters/rf_all_domain_eui_predictions.png)

   - **Error Distribution Plot**  
![eui_predictions_all_domain](/figures/model_plots/10_clusters/rf_all_domain_error_distribution.png)


The regional analysis shows that Asia & Oceania consistently demonstrates one of the best overall performances across validation strategies. This performance remains relatively stable in cross-domain validation, outperforming other regions. 

Central and South America also stands out with strong performance in both within-domain and cross-domain validation, though with more variable R² values. 

Other regions show more variable performance, with Europe, Northern America and Africa having higher error rates and less consistent R² values.


## 7. Conclusion and Discussion <a name="Conclusion"></a>

This project presents a robust and efficient methodology for estimating Energy Use Intensity (EUI) in buildings at a global scale, aiming to lay the groundwork for more accurate estimation of direct greenhouse gas (GHG) emissions from the building sector. Starting from a baseline KNN model that relied solely on latitude and longitude, we achieved a substantial improvement by incorporating climatic, geographic, and socioeconomic variables. This resulted in a reduction of the average error (MAPE) from 37.8% to 17.51% through the use of Random Forests. This 54% improvement in cross-domain validation highlights the model’s robustness and the importance of effective feature selection.

Among all input features, the most influential variables—based on feature importance from the Random Forest model—were the Income Index (32.09%), Average Temperature (23.58%), Latitude (16.41%), and Average Dewpoint Temperature (6.53%). These features consistently demonstrated strong predictive power regarding building energy use, highlighting the significant roles of economic development and climate in driving energy demand. Conversely, features such as GDP per capita (0.45%), the Paris Agreement indicator (0.01%), and the fuzzy cluster features derived from satellite imagery each contributed less than 1%. To improve model performance, we applied a threshold and retained only those features with importance greater than 1%, which led to a reduction in MAPE.

Future research could build on this approach to predict greenhouse gas (GHG) emissions from buildings by combining the estimated Energy Use Intensity (EUI) with building area data. This could lead to more accurate GHG estimates and help better understand the environmental impact of buildings.

The results of this project, however, may have been affected by the quality of the satellite images. For eight cities, satellite image data were missing entirely, and among the successfully retrieved images, the quality varied significantly from city to city. Identified issues included heavy coverage by unknown objects, significant blurriness, and, in some cases, retrieval of only about half of the expected images. Expanding the dataset to include higher-resolution and more recent satellite imagery, implementing data augmentation strategies, or integrating multi-modal data (such as combining satellite imagery with weather or geographic information) could further enhance the robustness and accuracy of the embedding and downstream predictions.

The embedding process also presents important opportunities for future improvement. The current method, based on the Clay model, produces generalized embeddings that may not fully capture task-specific characteristics. Future work could also explore on fine-tuning the embedding model for the specific downstream tasks, adopting domain adaptation techniques, or exploring alternative models specialized for geographic or environmental image analysis. 

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
│   │   ├── HDD_matchedscale.csv
│   │   ├── eui_boston.csv
│   │   ├── eui_california.csv
│   │   ├── eui_chicago.csv
│   │   ├── eui_miami.csv
│   │   ├── eui_nyc.csv
│   │   ├── eui_philadelphia.csv
│   │   ├── eui_seattle.csv
│   │   ├── eui_usa_cities_grouped_df.csv
│   │   ├── image_embeddings_matrix.csv
│   │   ├── image_results.csv
│   │   ├── image_results_v2.csv
│   │   ├── population_density.csv
│   │   └── temperature_dewpoint_precipitation_2023.csv
│   └── 03_processed
│       ├── merged_df.csv
│       ├── train_test_split_new_data.csv
│       └── train_test_split_original_data.csv
├── deliverables_agreement
│   ├── Capstone_Spring_Semester_Plan.pdf
│   └── Mid-Point Deliverables - Climate Trace.pdf
├── figures
│   ├── 01_formula.png
│   ├── 02_eui_map.png
│   ├── 03_region_map.png
│   ├── 04_experimental_design.png
│   ├── 05_fuzzy_partition_coefficient.png
│   ├── 06_feature_importance.png
│   ├── 06_feature_importance_10_clusters.png
│   ├── 06_feature_importance_20_clusters.png
│   ├── 06_feature_importance_5_clusters.png
│   ├── 07_avg_rf.png
│   ├── methodology-1.jpg
│   ├── methodology-2.jpg
│   ├── methodology-4.jpg
│   ├── methodology-5.jpg
│   ├── model_plots
│   │   ├── 10_clusters
│   │   ├── 20_clusters
│   │   └── 5_clusters
│   └── satteliteimage.jpg
├── notebooks
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
│   ├── 023_HDDPreprocessing.ipynb
│   ├── 024_CDDPreprocessing.ipynb
│   ├── 025_Population.ipynb
│   ├── 026_ImageEmbedding.ipynb
│   ├── 027_ImageEmbedding_Clustering.ipynb
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
│   └── catboost_info
│       ├── catboost_training.json
│       ├── learn
│       │   └── events.out.tfevents
│       ├── learn_error.tsv
│       ├── time_left.tsv
│       └── tmp
├── requirements.txt
├── results
│   ├── 10_clusters
│   ├── 20_clusters
│   ├── 5_clusters
│   └── gridsearch
│       ├── 20250428_1503_xgb_grid_search_results.csv
│       ├── 20250428_1504_cat_grid_search_results.csv
│       └── 20250428_1716_rf_grid_search_results.csv
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
