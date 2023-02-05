# 1. Import revelent data
# Import pandas
import pandas as pd

# Load dataset
cc_apps = pd.read_csv(r"C:\Users\kanem\Documents\MLcreditapp\crx.csv", header=None)

# Inspect data
cc_apps.head()

# 2. Understanding stucture and data info
# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print('\n')

# Print DataFrame information
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print('\n')

# Inspect missing values in the dataset
print(cc_apps.tail(17))

# 3. Splitting the data into train and test models
# Import train_test_split
from sklearn.model_selection import train_test_split

# Drop the features 11 and 13
cc_apps = cc_apps.drop([11, 13], axis=1)

# Split into train and test sets
cc_apps_train, cc_apps_test = train_test_split(cc_apps, test_size=0.33, random_state=42)