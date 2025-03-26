from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def get_train_test_split(merged_df, region, strategy, features):
    """
    Get training and testing data based on the selected strategy
    """
    if strategy == 'within_domain':
        region_df = merged_df[merged_df['Region Grouped'] == region]
        train_df = region_df[region_df['is_train'] == 1]
        test_df = region_df[region_df['is_train'] == 0]
    
    elif strategy == 'cross_domain':
        train_df = merged_df[
            (merged_df['Region Grouped'] != region) & 
            (merged_df['is_train'] == 1)
        ]
        test_df = merged_df[
            (merged_df['Region Grouped'] == region) & 
            (merged_df['is_train'] == 0)
        ]
    
    elif strategy == 'all_domain':
        train_df = merged_df[merged_df['is_train'] == 1]
        test_df = merged_df[
            (merged_df['Region Grouped'] == region) & 
            (merged_df['is_train'] == 0)
        ]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return {
        'X_train': train_df[features] if train_df is not None else None,
        'X_test': test_df[features],
        'y_train_res': train_df['Residential EUI (kWh/m2/year)'] if train_df is not None else None,
        'y_test_res': test_df['Residential EUI (kWh/m2/year)'],
        'y_train_non_res': train_df['Non-residential EUI (kWh/m2/year)'] if train_df is not None else None,
        'y_test_non_res': test_df['Non-residential EUI (kWh/m2/year)']
    }


def get_trained_model(X_train, X_test, y_train, y_test, model):
    """
    Train and get predictions from a provided model
    
    Parameters:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        model: sklearn estimator object
    
    Returns:
        Dictionary containing test values, predictions and trained model
    """
    # Add scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with scaled data
    model.fit(X_train_scaled, y_train)
    y_pred = pd.Series(
        model.predict(X_test_scaled), 
        index=y_test.index,
        name=y_test.name   
    )
    
    return {
        'y_test': y_test,
        'y_pred': y_pred,
        'model': model
    }


def train_and_evaluate_models(merged_df, regions, features, model, strategy):
    """
    Main function to train and evaluate models using different strategies
    
    Parameters:
        merged_df: DataFrame
        regions: list of regions
        features: list of feature names
        model: sklearn estimator object
        strategy: str ('within_domain', 'cross_domain', 'all_domain')
    
    Returns:
        dict: Dictionary containing results for each region and building type
    """
    results = {}
    
    for region in regions:
        data = get_train_test_split(merged_df, region, strategy, features)
        
        res_results = get_trained_model(
            data['X_train'], data['X_test'], 
            data['y_train_res'], data['y_test_res'], 
            model
        )
        non_res_results = get_trained_model(
            data['X_train'], data['X_test'], 
            data['y_train_non_res'], data['y_test_non_res'], 
            model
        )
        
        results[region] = {
            'residential': res_results,
            'non_residential': non_res_results
        }
    
    return results

def create_eui_comparison_plots(model_results, save_path=None):
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()

    for i, (region, result) in enumerate(model_results.items()):
        for j, building_type in enumerate(['residential', 'non_residential']):
            ax = axes[i + (5 if j == 1 else 0)]
            data = result[building_type]
            
            # Calculate metrics
            r2 = r2_score(data['y_test'], data['y_pred'])
            mape = np.mean(np.abs((data['y_test'] - data['y_pred']) / data['y_test'])) * 100
            
            # Main scatter plot
            ax.scatter(data['y_test'], data['y_pred'], alpha=0.7, s=20, color="#4F4F4F")
            ax.plot([data['y_test'].min(), data['y_test'].max()], 
                   [data['y_test'].min(), data['y_test'].max()], 'r--', lw=2)
            ax.plot([0, 400], [0, 400], 'g-', lw=2)
            
            # Labels and title
            building_label = 'Non-residential' if j == 1 else 'Residential'
            ax.set_xlabel(f'True {building_label} EUI', fontsize=14)
            ax.set_ylabel(f'Predicted {building_label} EUI', fontsize=14)
            ax.set_title(f'{region} ({building_label})\n'
                        f'R²: {r2:.2f}\n'
                        f'MAPE: {mape:.1f}%', 
                        fontsize=17)
            
            # Error bands
            x = np.linspace(0, 400, 401)
            
            # Add error bands individually with labels
            ax.plot(x, x * (1 + 0.10), '--', color='lightgreen', alpha=0.5, label='+/- 10% Error')
            ax.plot(x, x * (1 - 0.10), '--', color='lightgreen', alpha=0.5)
            
            ax.plot(x, x * (1 + 0.20), '--', color='orange', alpha=0.5, label='+/- 20% Error')
            ax.plot(x, x * (1 - 0.20), '--', color='orange', alpha=0.5)
            
            ax.plot(x, x * (1 + 0.30), '--', color='darkorange', alpha=0.5, label='+/- 30% Error')
            ax.plot(x, x * (1 - 0.30), '--', color='darkorange', alpha=0.5)
            
            ax.plot(x, x * (1 + 0.40), '--', color='red', alpha=0.5, label='+/- 40% Error')
            ax.plot(x, x * (1 - 0.40), '--', color='red', alpha=0.5)
            
            # Set axis limits and aspect ratio
            ax.set_xlim(0, 400)
            ax.set_ylim(0, 400)
            ax.set_aspect('equal', adjustable='box')
            
            # Add legend
            ax.legend(loc='upper left')
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model_strategies(merged_df, regions, features, model_type, feature_abbreviations, model_details=""):
   """
   Evaluate model performance across different regions and strategies, calculating various metrics.

   Parameters:
       merged_df: DataFrame with input data
       regions: list of geographic regions
       features: list of feature names for model training
       model_type: scikit-learn model class to use
       feature_abbreviations: dict mapping feature names to abbreviations
       model_details: str with additional model information (default "")
       
   Returns:
       DataFrame with performance metrics for each region, target (residential/non-residential), 
       and strategy (within/cross/all domain)
   """
   strategies = ['within_domain', 'cross_domain', 'all_domain']
   results_list = []
   
   for strategy in strategies:
       model = model_type()
       results = train_and_evaluate_models(
           merged_df=merged_df,
           regions=regions,
           features=features,
           model=model,
           strategy=strategy
       )
       
       for region in regions:
           for building_type in ['residential', 'non_residential']:
               data = results[region][building_type]
               r2 = r2_score(data['y_test'], data['y_pred'])
               mse = mean_squared_error(data['y_test'], data['y_pred'])
               mae = mean_absolute_error(data['y_test'], data['y_pred'])
               rmse = np.sqrt(mse)
               mape = np.mean(np.abs((data['y_test'] - data['y_pred']) / data['y_test'])) * 100
               wape = np.sum(np.abs(data['y_test'] - data['y_pred'])) / np.sum(np.abs(data['y_test'])) * 100
               
               results_list.append({
                   'Region': region,
                   'Target': 'Residential EUI (kWh/m2/year)' if building_type == 'residential' 
                            else 'Non-residential EUI (kWh/m2/year)',
                   'Strategy': strategy,
                   'Model': model_type.__name__,
                   'Model Details': model_details,
                   'Features Used': ', '.join(features),
                   'Features Abbreviated': ' | '.join(feature_abbreviations[f] for f in features),
                   'MSE': round(mse, 3),
                   'RMSE': round(rmse, 3),
                   'MAE': round(mae, 3),
                   'R²': round(r2, 3),
                   'MAPE': round(mape, 1),
                   'WAPE': round(wape, 1)
               })
   
   return pd.DataFrame(results_list)

def calculate_average_metrics(results_df):
   """Calculate average metrics by Target, Strategy and Model, keeping feature columns"""
   strategy_order = ['within_domain', 'cross_domain', 'all_domain']
   
   return results_df.groupby(['Target', 'Strategy', 'Model', 'Model Details', 'Features Used', 'Features Abbreviated']).agg({
       'MAPE': 'mean',
       'R²': 'mean',
       'MSE': 'mean',
       'RMSE': 'mean',
       'MAE': 'mean',
       'WAPE': 'mean'
   }).round(3).reindex(level='Strategy', labels=strategy_order)


def create_error_distribution_plots(model_results, save_path=None, binwidth=10):
    """
Create plots of the error distribution for each region and building type.

Parameters:
    model_results (dict): A dictionary containing the results for each region and building type.
    save_path (str, optional): The path to save the plot. If not provided, the plot will be displayed.
    binwidth (int, optional): The width of the histogram bins. Defaults to 10.
"""
    fig, axes = plt.subplots(2, 5, figsize=(23, 9))
    axes = axes.flatten()

    max_error = 150
    max_y = 25

    for i, (region, result) in enumerate(model_results.items()):
        for j, building_type in enumerate(['residential', 'non_residential']):
            ax = axes[i + (5 if j == 1 else 0)]
            data = result[building_type]
            errors = data['y_test'] - data['y_pred']
            
            # Add a small amount of noise if there is very little variation
            if len(np.unique(errors)) <= 5:
                errors = errors + np.random.normal(0, 0.01, size=errors.shape)
            
            max_error = max(max_error, abs(errors.min()), abs(errors.max()))
            
            # Ensure binwidth is positive
            binwidth = max(1, binwidth)
            
            # Calculate range and ensure it's positive (at least 0.1)
            range_value = max(0.1, errors.max() - errors.min())
            
            # Calculate number of bins and ensure it's at least 1
            bins = max(1, int(range_value / binwidth))
            
            # Use the calculated number of bins
            sns.histplot(errors, ax=ax, kde=True, bins=bins)
            
            building_label = 'Non-residential' if j == 1 else 'Residential'
            ax.set_xlabel(f'Errors ({building_label})', fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
            ax.set_title(f'{region} ({building_label})', fontsize=17)
            
            max_y = max(max_y, ax.get_ylim()[1])

    for ax in axes:
        ax.set_xlim(-max_error, max_error)
        ax.set_ylim(0, max_y)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


## GRID SEARCH 
def grid_search_best_params(merged_df, regions, features, model_class, param_grid, feature_abbreviations):
    """
    Performs grid search to find the best hyperparameters for a given model by evaluating
    multiple parameter combinations across different strategies, regions, and building types.
    Uses cross-validation on training data only.
    
    Parameters:
        merged_df: DataFrame with input data
        regions: list of geographic regions to evaluate
        features: list of feature names for model training
        model_class: scikit-learn model class to use
        param_grid: dictionary of parameters to search, with parameter names as keys and lists of values as values
        feature_abbreviations: dict mapping feature names to abbreviations
        
    Returns:
        tuple: (best_params, best_score) where:
            - best_params: dictionary with the best parameter combination
            - best_score: float representing the best average MAPE score (lower is better)
    """
    import itertools
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    
    train_df = merged_df.copy()
    train_df = train_df[train_df['is_train'] == 1]
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    strategies = ['within_domain', 'cross_domain', 'all_domain']
    
    # Initialize variables to track best parameters
    best_score = float('inf')
    best_params = None
    
    # Create KFold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Iterate through each parameter combination
    for combo_idx, param_combo in enumerate(param_combinations):
        current_params = dict(zip(param_names, param_combo))
        print(f"Evaluating combination {combo_idx+1}/{len(param_combinations)}: {current_params}")
        
        # Store CV MAPE scores for this parameter combination
        all_cv_mapes = []
        
        # For each fold in the cross-validation
        for train_idx, val_idx in kf.split(train_df):
            # Split the training data into train and validation for this fold
            cv_train_df = train_df.iloc[train_idx].copy()
            cv_val_df = train_df.iloc[val_idx].copy()
            
            # Set is_train flags for train_and_evaluate_models to work properly
            cv_train_df['is_train'] = 1
            cv_val_df['is_train'] = 0
            
            # Combine the CV train and validation sets
            cv_merged_df = pd.concat([cv_train_df, cv_val_df])
            
            # Evaluate each strategy using train_and_evaluate_models
            for strategy in strategies:
                # Initialize model with current parameters
                model = model_class(**current_params)
                
                # Use existing function to train and evaluate the model
                results = train_and_evaluate_models(
                    merged_df=cv_merged_df,
                    regions=regions,
                    features=features,
                    model=model,
                    strategy=strategy
                )
                
                # Collect MAPE scores for each region and building type
                for region in regions:
                    for building_type in ['residential', 'non_residential']:
                        # Some combinations might not have data, so check if the region exists
                        if region in results and building_type in results[region]:
                            data = results[region][building_type]
                            # Calculate MAPE
                            mape = np.mean(np.abs((data['y_test'] - data['y_pred']) / data['y_test'])) * 100
                            all_cv_mapes.append(mape)
        
        # Calculate average MAPE across all folds, strategies, regions and building types
        if all_cv_mapes:
            avg_mape = np.mean(all_cv_mapes)
            print(f"Average MAPE across all strategies/regions/building types: {avg_mape:.2f}%")
            
            # Update best parameters if current combination is better
            if avg_mape < best_score:
                best_score = avg_mape
                best_params = current_params
    
    print("\n=== Best Parameters ===")
    print(best_params)
    print(f"Average CV MAPE: {best_score:.2f}%")
    
    return best_params, best_score