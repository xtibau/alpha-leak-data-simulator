import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wntr

def plot_junction_pressure_vs_elevation(df, wn):
    """
    Plot mean pressure vs elevation for all junctions in the dataframe
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe containing pressure data with column names starting with "JUNCTION-"
    wn : wntr.network.WaterNetworkModel
        WNTR network model object containing elevation data
    
    Returns:
    --------
    result_df : pandas DataFrame
        DataFrame containing junction names, elevations, and mean pressures
    """
    # 1. Filter columns that start with 'JUNCTION'
    junction_columns = [col for col in df.columns if col.startswith('JUNCTION')]
    print(f"Found {len(junction_columns)} junction columns")
    
    if len(junction_columns) == 0:
        print("No junction columns found in dataframe")
        return pd.DataFrame()
    
    # 2. Calculate mean pressure for each junction
    mean_pressures = df[junction_columns].mean()
    
    # 3. Use the FULL column names directly (don't extract IDs)
    junction_names = junction_columns
    
    # Print some examples of column names for debugging
    print("\nExample column names (will use these directly):")
    for col in junction_names[:5]:
        print(f"  {col}")
    if len(junction_names) > 5:
        print(f"  ... ({len(junction_names) - 5} more)")
    
    # 4. Get elevation (altitude) data from WNTR network object
    elevations = {}
    missing_junctions = []
    
    # Get all node names from the WNTR model for debugging
    all_wntr_nodes = list(wn.node_name_list)
    print(f"\nWNTR model has {len(all_wntr_nodes)} nodes")
    print(f"First 5 node IDs in WNTR model: {all_wntr_nodes[:5] if len(all_wntr_nodes) >= 5 else all_wntr_nodes}")
    
    # Find matching nodes in the WNTR model
    for junction_name in junction_names:
        # Try direct match with full junction name
        if junction_name in all_wntr_nodes:
            try:
                node = wn.get_node(junction_name)
                elevations[junction_name] = node.elevation
                continue
            except Exception as e:
                print(f"Error accessing node {junction_name}: {e}")
        
        # Record missing junction if direct match failed
        missing_junctions.append(junction_name)
        elevations[junction_name] = np.nan
    
    # Report on missing junctions
    if missing_junctions:
        print(f"Warning: {len(missing_junctions)} junctions not found in network model")
        if len(missing_junctions) <= 10:
            print(f"Missing junctions: {missing_junctions}")
        else:
            print(f"First 10 missing junctions: {missing_junctions[:10]}...")
            
        # Find some example IDs that do exist in the network
        matched_junctions = [j_name for j_name in junction_names if j_name not in missing_junctions]
        if matched_junctions:
            print(f"Examples of junctions that were found: {matched_junctions[:5]}")
    
    # 5. Create a new dataframe with junction names, elevations, and mean pressures
    result_df = pd.DataFrame({
        'JunctionName': junction_names,
        'Elevation': [elevations.get(j_name, np.nan) for j_name in junction_names],
        'MeanPressure': mean_pressures.values
    })
    
    # 6. Remove any rows with missing elevation data
    result_df_clean = result_df.dropna(subset=['Elevation'])
    if len(result_df_clean) < len(result_df):
        print(f"Removed {len(result_df) - len(result_df_clean)} junctions with missing elevation data")
        result_df = result_df_clean
    
    # 7. Check if we have any valid data points
    if len(result_df) == 0:
        print("No valid junctions with elevation data found. Cannot create plot.")
        return result_df
        
    # Sort by elevation for better visualization
    result_df = result_df.sort_values('Elevation')
    
    # 8. Create the plot
    plt.figure(figsize=(12, 7))
    plt.scatter(result_df['Elevation'], result_df['MeanPressure'], 
               color='blue', alpha=0.6, s=50)
    
    # Add a trend line only if we have enough data points
    if len(result_df) > 1:  # Need at least 2 points for a line
        try:
            z = np.polyfit(result_df['Elevation'], result_df['MeanPressure'], 1)
            p = np.poly1d(z)
            plt.plot(result_df['Elevation'], p(result_df['Elevation']), 
                    "r--", linewidth=2, label=f'Trend: y={z[0]:.6f}x+{z[1]:.2f}')
            plt.legend()
        except (TypeError, np.linalg.LinAlgError) as e:
            print(f"Could not create trend line: {e}")
    
    # Add labels and title
    plt.xlabel('Elevation (m)')
    plt.ylabel('Mean Pressure (m)')
    plt.title('Junction Mean Pressure vs Elevation')
    plt.grid(True, alpha=0.3)
    
    # If the elevations span a wide range, consider using a log scale
    if len(result_df) > 1:  # Need at least 2 points to determine range
        elevation_range = result_df['Elevation'].max() - result_df['Elevation'].min()
        if elevation_range > 100:
            if result_df['Elevation'].min() > 0:  # Can only use log scale for positive values
                plt.xscale('log')
                plt.xlabel('Elevation (m, log scale)')
    
    plt.tight_layout()
    plt.show()
    
    # 9. Print summary statistics
    print("\nSummary statistics for junction mean pressures:")
    print(result_df['MeanPressure'].describe())
    print("\nSummary statistics for junction elevations:")
    print(result_df['Elevation'].describe())
    
    # 10. Additional Analysis: Calculate correlation between elevation and pressure
    if len(result_df) > 1:  # Need at least 2 points for correlation
        correlation = result_df['Elevation'].corr(result_df['MeanPressure'])
        print(f"\nCorrelation between Elevation and Mean Pressure: {correlation:.4f}")
    else:
        print("\nNot enough data points to calculate correlation.")
    
    return result_df

# Additional function in case of name mismatch between dataframe and WNTR model
def debug_node_name_matching(df, wn):
    """
    Debug function to help match dataframe column names with WNTR node names
    
    Parameters:
    -----------
    df : pandas DataFrame
        Dataframe containing pressure data
    wn : wntr.network.WaterNetworkModel
        WNTR network model object
    """
    # Get all column names from dataframe
    df_columns = list(df.columns)
    print(f"Dataframe has {len(df_columns)} columns")
    print(f"First 5 column names: {df_columns[:5]}")
    
    # Get all node names from WNTR model
    wntr_nodes = list(wn.node_name_list)
    print(f"WNTR model has {len(wntr_nodes)} nodes")
    print(f"First 5 node names: {wntr_nodes[:5]}")
    
    # Check for any direct matches
    direct_matches = set(df_columns).intersection(set(wntr_nodes))
    print(f"Found {len(direct_matches)} direct matches between dataframe columns and WNTR nodes")
    if direct_matches:
        print(f"Examples of matching names: {list(direct_matches)[:5]}")
    
    # If no direct matches, suggest potential mapping approaches
    if not direct_matches:
        print("\nNo direct matches found. Suggesting potential approaches:")
        
        # Check if dataframe columns might be prefixed versions of WNTR nodes
        if df_columns and wntr_nodes:
            df_col_sample = df_columns[0]
            wntr_node_sample = wntr_nodes[0]
            print(f"Example dataframe column: '{df_col_sample}'")
            print(f"Example WNTR node: '{wntr_node_sample}'")
            
            if '-' in df_col_sample:
                prefix, id_part = df_col_sample.split('-', 1)
                print(f"  Dataframe column seems to have format: {prefix}-{id_part}")
                if id_part in wntr_nodes:
                    print(f"  Found match using ID part only: '{id_part}' matches a WNTR node")
                    print("  Suggestion: Use the ID part after the hyphen to match with WNTR nodes")
            
            # Check case sensitivity
            print("\nChecking case sensitivity:")
            df_cols_lower = [col.lower() for col in df_columns[:10]]
            wntr_nodes_lower = [node.lower() for node in wntr_nodes[:20]]
            case_insensitive_matches = set(df_cols_lower).intersection(set(wntr_nodes_lower))
            if case_insensitive_matches:
                print(f"  Found {len(case_insensitive_matches)} case-insensitive matches")
                print("  Suggestion: Try matching names ignoring case")
        
        print("\nPossible solution approaches:")
        print("1. Extract relevant parts from dataframe column names to match WNTR nodes")
        print("2. Map dataframe columns to WNTR nodes using a custom mapping function")
        print("3. Update WNTR model to use same naming convention as dataframe")