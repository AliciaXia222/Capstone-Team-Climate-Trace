from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

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
    model.fit(X_train, y_train)
    y_pred = pd.Series(
        model.predict(X_test), 
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