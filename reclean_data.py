import os
import pandas as pd

def delete_columns_from_csv(input_file, output_file, start_col, end_col):
    """
    Deletes columns from index `start_col` to `end_col` (inclusive) in a CSV file.
    
    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file.
        start_col (int): Starting column index (0-based).
        end_col (int): Ending column index (0-based).
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Drop the specified columns
    df = df.drop(df.columns[start_col:end_col + 1], axis=1)
    
    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the modified DataFrame to the output file
    df.to_csv(output_file, index=False)
    print(f"Processed file saved: {output_file}")

def process_folder(input_folder, output_folder, start_col, end_col):
    """
    Processes all CSV files in the input folder and its subfolders.
    Deletes specified columns and saves the modified files in the output folder.
    
    Parameters:
        input_folder (str): Path to the input folder containing CSV files.
        output_folder (str): Path to the output folder for processed files.
        start_col (int): Starting column index (0-based).
        end_col (int): Ending column index (0-based).
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".csv"):
                # Construct full paths for input and output files
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_file = os.path.join(output_folder, relative_path, file)
                
                # Process the CSV file
                delete_columns_from_csv(input_file, output_file, start_col, end_col)

# Example usage
input_folder = "input_folder"      # Path to the folder with original CSV files
output_folder = "output_folder"    # Path to the folder for saving modified files
start_column = 2                   # Start index (0-based)
end_column = 4                     # End index (0-based)

process_folder('./SisFall_split', './SisFall_recleaned', 4, 9)
process_folder('./SisFall_recleaned', './SisFall_recleaned', 0, 0)