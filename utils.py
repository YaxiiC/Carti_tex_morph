
import os
import psutil
import pandas as pd

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory used: {mem_info.rss / 1024 ** 2:.2f} MB")



def save_model_weights(model, feature_names, output_file="feature_weights.csv"):
    """
    Save the weights of a trained linear regression model to a CSV file.
    """
    weights = model.coef_
    df = pd.DataFrame(data=[weights], columns=feature_names)
    df.to_csv(output_file, index=False)
    print(f"Feature weights saved to {output_file}")
