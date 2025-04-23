#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import glob
import numpy as np

def combine_csv_files(input_dir, output_file, summary_file=None):
    """
    Combines all CSV files from the input directory and its subdirectories
    into a single CSV file, adding the folder structure as columns.
    
    Args:
        input_dir (str): Path to the root directory containing CSV files
        output_file (str): Path to save the combined CSV file
        summary_file (str): Optional path to save summary statistics
    """
    print(f"Looking for CSV files in {input_dir} and its subdirectories...")
    
    # List to store all dataframes
    all_dfs = []
    skipped_files = []
    
    # Walk through directory structure
    for root, dirs, files in os.walk(input_dir):
        # Skip hidden folders
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Find CSV files in current directory
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if not csv_files:
            continue
            
        for csv_file in csv_files:
            file_path = os.path.join(root, csv_file)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Skip empty files
                if df.empty:
                    print(f"Skipping empty file: {file_path}")
                    skipped_files.append((file_path, "Empty file"))
                    continue
                
                # Get relative path from input_dir
                rel_path = os.path.relpath(root, input_dir)
                
                # Split the path to extract meaningful parts
                path_parts = rel_path.split(os.sep)
                
                # Determine how many parts we have and assign appropriately
                # For a structure like Person/JumpType/Surface
                person = path_parts[0] if len(path_parts) > 0 else "Unknown"
                jump_type = path_parts[1] if len(path_parts) > 1 else "Unknown"
                surface = path_parts[2] if len(path_parts) > 2 else "Unknown"
                
                # Add filename without extension as Trial number
                trial = os.path.splitext(csv_file)[0]
                
                # Add metadata columns
                df['Person'] = person
                df['JumpType'] = jump_type
                df['Surface'] = surface
                df['Trial'] = trial
                df['FilePath'] = file_path  # Include original file path for reference
                
                # Append to our list
                all_dfs.append(df)
                print(f"Added: {file_path}")
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                skipped_files.append((file_path, str(e)))
    
    if not all_dfs:
        print("No CSV files found or all files were skipped!")
        return
    
    # Combine all dataframes
    print("\nCombining dataframes...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Convert numeric columns that might have been read as strings
    numeric_cols = [
        'Calculated Jump Height (cm)', 
        'Calculated Flight Time (s)', 
        'Take-off Velocity (est.) (m/s)',
        'Knee Angle (at lowest point) (degrees)',
        'Hip Angle (at peak) (degrees)'
    ]
    
    for col in numeric_cols:
        if col in combined_df.columns:
            try:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            except:
                print(f"Warning: Could not convert column '{col}' to numeric")
    
    # Reorder columns to put metadata first
    meta_cols = ['Person', 'JumpType', 'Surface', 'Trial', 'Status', 'FilePath']
    other_cols = [col for col in combined_df.columns if col not in meta_cols]
    combined_df = combined_df[meta_cols + other_cols]
    
    # Save to output file
    combined_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully combined {len(all_dfs)} CSV files into {output_file}")
    print(f"Total rows in combined file: {len(combined_df)}")
    
    # Calculate and display summary statistics
    print("\n--- Summary Statistics ---")
    success_count = combined_df[combined_df['Status'] == 'Success'].shape[0]
    failed_count = combined_df[combined_df['Status'] == 'Failed'].shape[0]
    
    print(f"Successful analyses: {success_count}")
    print(f"Failed analyses: {failed_count}")
    
    if 'Calculated Jump Height (cm)' in combined_df.columns:
        success_df = combined_df[combined_df['Status'] == 'Success']
        if not success_df.empty:
            avg_height = success_df['Calculated Jump Height (cm)'].mean()
            max_height = success_df['Calculated Jump Height (cm)'].max()
            min_height = success_df['Calculated Jump Height (cm)'].min()
            print(f"\nJump Height Statistics (successful jumps only):")
            print(f"  Average: {avg_height:.2f} cm")
            print(f"  Maximum: {max_height:.2f} cm")
            print(f"  Minimum: {min_height:.2f} cm")
    
    # Generate summary file if requested
    if summary_file:
        print(f"\nGenerating summary statistics to {summary_file}...")
        
        # Calculate statistics per person, jump type, and surface
        if 'Calculated Jump Height (cm)' in combined_df.columns:
            success_df = combined_df[combined_df['Status'] == 'Success']
            
            if not success_df.empty:
                # Group by different combinations and calculate statistics
                person_stats = success_df.groupby('Person')['Calculated Jump Height (cm)'].agg(['mean', 'std', 'min', 'max']).reset_index()
                person_stats.columns = ['Person', 'Mean Height (cm)', 'Std Dev (cm)', 'Min Height (cm)', 'Max Height (cm)']
                
                jump_type_stats = success_df.groupby('JumpType')['Calculated Jump Height (cm)'].agg(['mean', 'std', 'min', 'max']).reset_index()
                jump_type_stats.columns = ['Jump Type', 'Mean Height (cm)', 'Std Dev (cm)', 'Min Height (cm)', 'Max Height (cm)']
                
                surface_stats = success_df.groupby('Surface')['Calculated Jump Height (cm)'].agg(['mean', 'std', 'min', 'max']).reset_index()
                surface_stats.columns = ['Surface', 'Mean Height (cm)', 'Std Dev (cm)', 'Min Height (cm)', 'Max Height (cm)']
                
                person_jump_stats = success_df.groupby(['Person', 'JumpType'])['Calculated Jump Height (cm)'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
                person_jump_stats.columns = ['Person', 'Jump Type', 'Mean Height (cm)', 'Std Dev (cm)', 'Min Height (cm)', 'Max Height (cm)', 'Count']
                
                # Create summary DataFrames for skipped files if any
                if skipped_files:
                    skipped_df = pd.DataFrame(skipped_files, columns=['File Path', 'Error'])
                
                # Try to write to Excel, fall back to CSV if Excel writing fails
                try:
                    # Check if openpyxl is available
                    import openpyxl
                    
                    # If we get here, openpyxl is available
                    summary_base = os.path.splitext(summary_file)[0]
                    
                    with pd.ExcelWriter(summary_file) as writer:
                        person_stats.to_excel(writer, sheet_name='By Person', index=False)
                        jump_type_stats.to_excel(writer, sheet_name='By Jump Type', index=False)
                        surface_stats.to_excel(writer, sheet_name='By Surface', index=False)
                        person_jump_stats.to_excel(writer, sheet_name='Person & Jump Type', index=False)
                        
                        if skipped_files:
                            skipped_df.to_excel(writer, sheet_name='Skipped Files', index=False)
                    
                    print(f"Summary statistics saved to Excel file: {summary_file}")
                
                except (ImportError, ModuleNotFoundError) as e:
                    # openpyxl not available, fall back to CSV files
                    print(f"Warning: Could not create Excel file ({e}). Saving summary statistics as CSV files instead.")
                    
                    # Create directory for summary files if it doesn't exist
                    summary_dir = os.path.dirname(summary_file)
                    summary_base = os.path.splitext(os.path.basename(summary_file))[0]
                    if summary_dir and not os.path.exists(summary_dir):
                        os.makedirs(summary_dir, exist_ok=True)
                    
                    # Save each dataframe as a separate CSV file
                    person_csv = os.path.join(summary_dir, f"{summary_base}_by_person.csv") if summary_dir else f"{summary_base}_by_person.csv"
                    jump_type_csv = os.path.join(summary_dir, f"{summary_base}_by_jump_type.csv") if summary_dir else f"{summary_base}_by_jump_type.csv"
                    surface_csv = os.path.join(summary_dir, f"{summary_base}_by_surface.csv") if summary_dir else f"{summary_base}_by_surface.csv"
                    person_jump_csv = os.path.join(summary_dir, f"{summary_base}_by_person_jump.csv") if summary_dir else f"{summary_base}_by_person_jump.csv"
                    
                    person_stats.to_csv(person_csv, index=False)
                    jump_type_stats.to_csv(jump_type_csv, index=False)
                    surface_stats.to_csv(surface_csv, index=False)
                    person_jump_stats.to_csv(person_jump_csv, index=False)
                    
                    if skipped_files:
                        skipped_csv = os.path.join(summary_dir, f"{summary_base}_skipped_files.csv") if summary_dir else f"{summary_base}_skipped_files.csv"
                        skipped_df.to_csv(skipped_csv, index=False)
                    
                    print(f"Summary statistics saved as CSV files with prefix: {summary_base}_")
                
                except Exception as e:
                    print(f"Error saving summary statistics: {e}")
            else:
                print("No successful jump data to generate statistics.")
        else:
            print("Jump height column not found in data. Cannot generate statistics.")
    
    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine all CSV files from a directory structure into a single CSV file."
    )
    parser.add_argument(
        "--input_dir", 
        required=True,
        help="Path to the root directory containing CSV files (e.g., data/template/angle)"
    )
    parser.add_argument(
        "--output_file", 
        default="combined_results.csv",
        help="Path to save the combined CSV file (default: combined_results.csv)"
    )
    parser.add_argument(
        "--summary_file", 
        default="jump_summary.xlsx",
        help="Path to save summary statistics in Excel format (default: jump_summary.xlsx)"
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
        exit(1)
        
    combine_csv_files(args.input_dir, args.output_file, args.summary_file) 