from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd

# Load Dataset
energy_efficiency_data = fetch_ucirepo(id=242)

energy_df = energy_efficiency_data.data.original #Get the Pandas DataFrame

# Split the data, beginning by sampling 70% of the full dataset for training
training_set = energy_df.sample(frac=0.7, random_state=345345)
temp_set = energy_df.drop(training_set.index) # Create a temp set containing the remaing 30% of data
validation_set = temp_set.sample(frac=0.33, random_state=654168) # Take 33% of the remaining 30% (so 10% of original data)
test_set = temp_set.drop(validation_set.index) # Create the testing set by dropping the validation set from the temp set


print(energy_df.describe())
print(f'\n {training_set.describe()} \n{validation_set.describe()} \n{test_set.describe()}')
