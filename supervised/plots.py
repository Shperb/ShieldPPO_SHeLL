import pandas as pd


def process_predictions(input_csv, output_csv):
    """
    Processes the input CSV by rounding 'cost' and 'prediction' to 2 decimal places,
    calculates the number of correct predictions for each cost, and saves the result
    to a new CSV file.

    Parameters:
    - input_csv: str, path to the input CSV file with columns 'cost' and 'prediction'
    - output_csv: str, path where the output CSV will be saved
    """

    # Read the input CSV
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: The file {input_csv} does not exist.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {input_csv} is empty.")
        return
    except pd.errors.ParserError:
        print(f"Error: The file {input_csv} does not appear to be in CSV format.")
        return

    # Check if required columns exist
    if not {'cost', 'prediction'}.issubset(df.columns):
        print("Error: Input CSV must contain 'cost' and 'prediction' columns.")
        return


    # Step 1: Binarize 'cost' column
    # If cost is exactly 1, set to 1; else, set to 0
    df['binarized_costs'] = df['cost'].apply(lambda x: 1 if x == 1 else 0)

    # Step 2: Round 'prediction' to 2 decimal places
    df['rounded_predictions'] = df['prediction'].round(2)

    # Step 3: Aggregate counts
    # For each rounded_prediction, count the number of times cost was 0 and 1
    aggregation = df.groupby(['rounded_predictions', 'binarized_costs']).size().unstack(fill_value=0)

    # Ensure both columns for 0 and 1 exist
    if 0 not in aggregation.columns:
        aggregation[0] = 0
    if 1 not in aggregation.columns:
        aggregation[1] = 0

    # Rename the columns for clarity
    aggregation = aggregation.rename(columns={
        0: 'count_cost_0',
        1: 'count_cost_1'
    }).reset_index()

    # Optional: Sort by rounded_predictions for better readability
    aggregation = aggregation.sort_values(by='rounded_predictions')

    # # Rename columns to match the desired output
    # aggregated_df.rename(columns={
    #     'binarized_costs': 'rounded_costs',
    #     'rounded_predictions': 'rounded_predictions',
    #     'correct_prediction': 'correct_predictions'
    # }, inplace=True)

    # Save the aggregated data to a new CSV
    try:
        aggregation.to_csv(output_csv, index=False)
        print(f"Processed data has been saved to {output_csv}")
    except Exception as e:
        print(f"Error: Could not save to {output_csv}. {e}")


# Example usage:
if __name__ == "__main__":
    input_file = 'models/Kinematics_1733770798/costs_predictions_trial_0.csv'  # Replace with your input CSV file path
    output_file = input_file.replace('.csv', '_rounded.csv')  # Replace with your input CSV file path
    process_predictions(input_file, output_file)
