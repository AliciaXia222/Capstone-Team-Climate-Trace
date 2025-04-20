from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from glob import glob
import os


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



def create_metrics_comparison(results_path="../results"):
    """
    Create a comparison table for different models using all evaluation metrics.
    """
    
    # Metric configuration
    metric_config = {
        'WAPE': {'format': '{:.1f}%'},
        'MAPE': {'format': '{:.1f}%'},
        'R²': {'format': '{:.3f}'},
        'MSE': {'format': '{:.1f}'},
        'RMSE': {'format': '{:.1f}'},
        'MAE': {'format': '{:.0f}'},
    }
    
    # Load results
    dfs = []
    for file_path in glob(os.path.join(results_path, "*average_results.csv")):
        df = pd.read_csv(file_path, index_col=0)
        df['Source_File'] = os.path.basename(file_path)
        dfs.append(df)

    all_models_results = pd.concat(dfs, ignore_index=True)

    # Mapping dictionaries with defined order y agregando Lasso y Ridge
    model_names = {
        'LinearRegression': 'LR',
        'KNeighborsRegressor': 'KNN \n(Base line)',
        'RandomForestRegressor': 'RF',
        'XGBRegressor': 'XGB',
        'CatBoostRegressor': 'CAT',
        'LR Lasso': 'Lasso',   
        'LR Ridge': 'Ridge'    
    }
    
    strategy_names = {
        'within_domain': 'Within Domain',
        'cross_domain': 'Cross Domain',
        'all_domain': 'All Domain'
    }

    target_names = {
        'Non-residential EUI (kWh/m2/year)': 'Non-residential',
        'Residential EUI (kWh/m2/year)': 'Residential'
    }

    # Apply mappings
    all_models_results['Strategy'] = all_models_results['Strategy'].map(strategy_names)
    all_models_results['Model'] = all_models_results['Model'].map(model_names)
    all_models_results['Target'] = all_models_results['Target'].map(target_names)

    model_order = ['KNN \n(Base line)','LR', 'Lasso', 'Ridge', 'RF', 'XGB', 'CAT']
    all_models_results['Model'] = pd.Categorical(
        all_models_results['Model'],
        categories=model_order,
        ordered=True
    )

    # Create full detailed table with ordered metrics
    metrics = ['MAPE', 'R²', 'MSE', 'MAE', 'WAPE', 'RMSE']
    
    full_table = pd.pivot_table(
        all_models_results,
        index=['Model', 'Target', 'Model Details','Features Used','Features Abbreviated'],
        columns=['Strategy'],
        values=metrics,
        aggfunc='mean'
    ).sort_index(level=0)

    # Format the values according to metric configuration
    formatted_table = full_table.copy()
    for metric in metrics:
        for strategy in strategy_names.values():
            col = (metric, strategy)
            formatted_table[col] = formatted_table[col].apply(
                lambda x: metric_config[metric]['format'].format(x)
            )
    #df_flat = formatted_table.copy()
    #df_flat.columns = [f"{col[0]} {col[1]}" if col[1] != '' else col[0] for col in formatted_table.columns]

    return formatted_table

def create_model_comparison(comparison_table, metric, save_path=None, y_limits=None):
    """
    Create a comparison plot for different models using a specified metric.
    
    Parameters:
    -----------
    comparison_table : pd.DataFrame
        The formatted table from create_metrics_comparison function
    metric : str
        The metric to compare. Options: 'MAPE', 'R²', 'MSE', 'RMSE', 'MAE', 'WAPE'
    save_path : str
        The path where the plot should be saved
    y_limits : tuple, optional
        A tuple of (min, max) values to set the y-axis limits manually. 
        If None, limits will be set automatically based on the data and metric type.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot
    """
    
    # Metric configuration with average labels
    metric_config = {
        'R²': {'format': '%.2f', 'label': 'Avg. R²', 'limits': (-1, 1)},
        'MSE': {'format': '%.0f', 'label': 'Avg. MSE', 'limits': (0, None)},
        'RMSE': {'format': '%.1f', 'label': 'Avg. RMSE', 'limits': (0, None)},
        'MAE': {'format': '%.1f', 'label': 'Avg. MAE', 'limits': (0, None)},
        'WAPE': {'format': '%.1f%%', 'label': 'Avg. WAPE (%)', 'limits': (0, 100)},
        'MAPE': {'format': '%.0f%%', 'label': 'Avg. MAPE (%)', 'limits': (0, 100)}
    }
    
    # Prepare data for plotting
    plot_data = comparison_table[metric].reset_index()
    plot_data = plot_data.melt(id_vars=['Model', 'Target'], 
                              var_name='Strategy', 
                              value_name=metric)
    
    # Convert string percentages to float for plotting
    if metric in ['MAPE', 'WAPE']:
        plot_data[metric] = plot_data[metric].str.rstrip('%').astype(float)
    else:
        plot_data[metric] = plot_data[metric].astype(float)

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    strategies = ['Within Domain', 'Cross Domain', 'All Domain']

    # Set y-axis limits based on provided limits or calculate them dynamically
    if y_limits is not None:
        y_min, y_max = y_limits
    else:
        # Calculate dynamic limits
        min_val = plot_data[metric].min()
        max_val = plot_data[metric].max()
        padding = (max_val - min_val) * 0.1
        
        # Set y-axis limits based on metric configuration
        base_min, base_max = metric_config[metric]['limits']
        y_min = base_min if base_min is not None else max(min_val - padding, 0)
        y_max = base_max if base_max is not None else max_val + padding

    for ax, strategy in zip(axes, strategies):
        strategy_data = plot_data[plot_data['Strategy'] == strategy]
        
        # Get unique models and targets for custom coloring
        models = strategy_data['Model'].unique()
        targets = strategy_data['Target'].unique()
        
        # Create an empty list to store bar containers
        containers = []
        
        # Define regular color palette for all models
        regular_palette = {
            'Non-residential': '#fcaa70', 
            'Residential': '#84aaca'       
        }
        
        # Plot each target separately
        for target in targets:
            target_data = strategy_data[strategy_data['Target'] == target]
            
            # Create a list to store colors for each bar - now all use regular_palette
            colors = []
            for model in target_data['Model']:
                colors.append(regular_palette[target])
            
            # Create bar plot for this target
            bars = ax.bar(
                x=[i + (0.4 if target == targets[1] else 0) for i in range(len(models))],
                height=target_data[metric].values,
                width=0.4,
                label=target,
                color=colors
            )
            
            containers.append(bars)
            
            # Add value labels
            for i, bar in enumerate(bars):
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    bar.get_height() + 0.5,
                    f"{target_data[metric].iloc[i]:.0f}%" if metric in ['MAPE', 'WAPE'] else f"{target_data[metric].iloc[i]:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=13
                )
        
        # Set x-ticks at the center of grouped bars
        ax.set_xticks([i + 0.2 for i in range(len(models))])
        ax.set_xticklabels(models, fontsize=16)
        
        # Customize appearance
        ax.set_title(f'{strategy}',
                    fontsize=20,
                    pad=20,
                    fontweight='bold')
        ax.set_ylabel(metric_config[metric]['label'], fontsize=16)
        ax.set_ylim(y_min, y_max)
        
        # Remove x-axis label
        ax.set_xlabel('')
        
        # Add baseline
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        
        # Position legend with adjusted position and style
        if strategy == 'Cross Domain':
            # Create custom legend handles using only regular palette
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=regular_palette['Non-residential'], label='Non-residential'),
                Patch(facecolor=regular_palette['Residential'], label='Residential')
            ]
            
            legend = ax.legend(handles=legend_elements,
                             bbox_to_anchor=(0.5, -0.15),
                             loc='upper center',
                             ncol=2,
                             fontsize=16)
            # Remove bold from legend labels
            for text in legend.get_texts():
                text.set_fontweight('normal')
        else:
            ax.legend([],[], frameon=False)

    plt.tight_layout()

    # Save plot if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close any existing figures to prevent double plotting
    plt.show() 
    plt.close('all')
    
## GRID SEARCH 
def grid_search_best_params(merged_df, regions, features, model_class, param_grid, feature_abbreviations):
    """
    Performs grid search to find the best hyperparameters for a given model by evaluating
    multiple parameter combinations across different strategies and building types.
    Maximizes performance on cross_domain strategy only, but tracks all strategies.
    Uses cross-validation on training data only.
    
    Parameters:
        merged_df: DataFrame with input data
        regions: list of geographic regions to evaluate
        features: list of feature names for model training
        model_class: scikit-learn model class to use
        param_grid: dictionary of parameters to search, with parameter names as keys and lists of values as values
        feature_abbreviations: dict mapping feature names to abbreviations
        
    Returns:
        tuple: (best_params, best_score, all_results) where:
            - best_params: dictionary with the best parameter combination for cross_domain
            - best_score: float representing the best average MAPE score for cross_domain (lower is better)
            - all_results: dictionary with results for all parameter combinations, strategies and building types
    """
    import itertools
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    
    # Filter training data only
    train_df = merged_df.copy()
    train_df = train_df[train_df['is_train'] == 1]
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    strategies = ['within_domain', 'cross_domain', 'all_domain']
    building_types = ['residential', 'non_residential']
    
    # Initialize variables to track best parameters for cross_domain
    best_cross_domain_score = float('inf')
    best_cross_domain_params = None
    
    # Dictionary to store all results
    all_results = {}
    
    # Create KFold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Iterate through each parameter combination
    for combo_idx, param_combo in enumerate(param_combinations):
        current_params = dict(zip(param_names, param_combo))
        print(f"Evaluating combination {combo_idx+1}/{len(param_combinations)}: {current_params}")
        
        # Store building type specific MAPE scores (averaged across regions)
        # Structure: {strategy: {building_type: [mape values across folds and regions]}}
        strategy_bt_mapes = {
            strategy: {
                bt: [] for bt in building_types
            } for strategy in strategies
        }
        
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
                
                # Collect MAPE scores for each building type (across all regions)
                for region in regions:
                    if region in results:
                        for building_type in building_types:
                            if building_type in results[region]:
                                data = results[region][building_type]
                                # Calculate MAPE
                                mape = np.mean(np.abs((data['y_test'] - data['y_pred']) / data['y_test'])) * 100
                                strategy_bt_mapes[strategy][building_type].append(mape)
        
        # Calculate average MAPE across all folds and regions for each strategy and building type
        combo_results = {}
        for strategy in strategies:
            strategy_results = {'overall': 0.0, 'count': 0}
            
            for building_type in building_types:
                mape_values = strategy_bt_mapes[strategy][building_type]
                if mape_values:
                    avg_mape = np.mean(mape_values)
                    strategy_results[building_type] = avg_mape
                    
                    # Add to strategy overall
                    strategy_results['overall'] += avg_mape
                    strategy_results['count'] += 1
            
            # Calculate strategy overall average if we have data
            if strategy_results['count'] > 0:
                strategy_results['overall'] = strategy_results['overall'] / strategy_results['count']
                print(f"Strategy '{strategy}' - Overall Average MAPE: {strategy_results['overall']:.2f}%")
            
            combo_results[strategy] = strategy_results
        
        # Store all results for this parameter combination
        param_key = '_'.join([f"{k}={v}" for k, v in current_params.items()])
        all_results[param_key] = combo_results
        
        # Update best parameters for cross_domain if current combination is better
        if 'cross_domain' in combo_results and combo_results['cross_domain']['count'] > 0:
            cross_domain_mape = combo_results['cross_domain']['overall']
            if cross_domain_mape < best_cross_domain_score:
                best_cross_domain_score = cross_domain_mape
                best_cross_domain_params = current_params
    
    print("\n=== Best Parameters for cross_domain strategy ===")
    print(best_cross_domain_params)
    print(f"Cross-domain Average CV MAPE: {best_cross_domain_score:.2f}%")
    
    return best_cross_domain_params, best_cross_domain_score, all_results

def convert_results_to_dataframe(all_results, features=None):
    """
    Converts the grid search results dictionary to a pandas DataFrame.
    This version creates a simplified structure with a single column for parameters
    and an optional column for features used.
    
    Parameters:
        all_results: dictionary with results of all parameter combinations
        features: list of feature names used in the grid search (optional)
        
    Returns:
        pandas DataFrame with flattened results
    """
    import pandas as pd
    import json
    
    # Create a list to store flattened results
    flattened_results = []
    
    # Iterate through each parameter combination
    for param_idx, (param_key, strategy_results) in enumerate(all_results.items()):
        # Add combination index for reference
        base_params = {'combination_idx': param_idx}
        
        # Extract parameters from the key and store as a dictionary string
        try:
            # First attempt to parse the param_key into a proper dictionary
            param_dict = {}
            param_parts = param_key.split('_')
            for part in param_parts:
                if '=' in part:
                    name, value = part.split('=', 1)  # Split on first '=' only
                    
                    # Try to convert to appropriate data type
                    if value.isdigit():
                        param_dict[name] = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                        param_dict[name] = float(value)
                    elif value.lower() in ['true', 'false']:
                        param_dict[name] = value.lower() == 'true'
                    elif value.lower() in ['none', 'null']:
                        param_dict[name] = None
                    else:
                        param_dict[name] = value
            
            # Convert the dictionary to a string representation
            base_params['params'] = str(param_dict)
        except:
            # If parsing fails, use the raw param_key
            base_params['params'] = param_key
        
        # Add features column if provided
        if features is not None:
            base_params['features'] = str(features)
        
        # For each strategy in this parameter combination
        for strategy, strategy_data in strategy_results.items():
            if isinstance(strategy_data, dict):
                # Get overall strategy MAPE
                strategy_overall_mape = strategy_data.get('overall')
                
                # Create a base row with strategy info
                base_row = {
                    'strategy': strategy,
                    **base_params
                }
                
                if strategy_overall_mape is not None:
                    base_row['mape_overall'] = strategy_overall_mape
                
                # Add a row for the strategy overall
                overall_row = base_row.copy()
                overall_row['building_type'] = 'all'
                flattened_results.append(overall_row)
                
                # For each building type in this strategy
                for bt, bt_mape in strategy_data.items():
                    if bt not in ['overall', 'count'] and isinstance(bt_mape, (int, float)):
                        # Add a row for this specific building type
                        bt_row = base_row.copy()
                        bt_row['building_type'] = bt
                        bt_row['mape'] = bt_mape
                        flattened_results.append(bt_row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(flattened_results)
    
    # Sort columns for better readability if DataFrame is not empty
    if not results_df.empty:
        # Define column order
        id_cols = ['strategy', 'building_type']
        metric_cols = [col for col in results_df.columns if 'mape' in col.lower()]
        info_cols = ['params']
        if 'features' in results_df.columns:
            info_cols.append('features')
        ref_cols = ['combination_idx']
        
        # Combine all columns in the desired order
        sorted_cols = id_cols + metric_cols + info_cols + ref_cols
        
        # Only include columns that exist in the DataFrame
        existing_cols = [col for col in sorted_cols if col in results_df.columns]
        results_df = results_df[existing_cols]
    
    return results_df