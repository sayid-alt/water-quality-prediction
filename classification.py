# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # **Water Quality Prediction for Public Health Protection**
# 
# ## **Goal**
# 
# The primary goal of this project is to protect public health by identifying unsafe water sources to help prevent waterborne diseases. By leveraging machine learning techniques, the project aims to build a predictive model that determines whether water is potable (safe for consumption) or non-potable (unsafe).
# 
# >**This project has the potential to contribute meaningfully to public health protection, ensuring safer water supplies for communities worldwide.**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:12:53.838537Z","iopub.execute_input":"2024-12-16T12:12:53.839000Z","iopub.status.idle":"2024-12-16T12:12:55.000060Z","shell.execute_reply.started":"2024-12-16T12:12:53.838964Z","shell.execute_reply":"2024-12-16T12:12:54.998615Z"}}
!python --version

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:33:10.883595Z","iopub.execute_input":"2024-12-16T12:33:10.884111Z","iopub.status.idle":"2024-12-16T12:33:10.891230Z","shell.execute_reply.started":"2024-12-16T12:33:10.884057Z","shell.execute_reply":"2024-12-16T12:33:10.889493Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.plotting import scatter_matrix

from IPython.display import display
import os

import warnings
warnings.filterwarnings("ignore")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # **Data Loading**

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:13:25.432222Z","iopub.execute_input":"2024-12-16T12:13:25.432602Z","iopub.status.idle":"2024-12-16T12:13:25.497465Z","shell.execute_reply.started":"2024-12-16T12:13:25.432569Z","shell.execute_reply":"2024-12-16T12:13:25.496388Z"},"jupyter":{"outputs_hidden":false}}
INPUT_DIR = '/kaggle/input/water-potability/'
WORKING_DIR = '/kaggle/working/'

input_dataset = os.path.join(INPUT_DIR, 'water_potability.csv')
df = pd.read_csv(input_dataset)
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:13:30.099493Z","iopub.execute_input":"2024-12-16T12:13:30.099969Z","iopub.status.idle":"2024-12-16T12:13:30.162548Z","shell.execute_reply.started":"2024-12-16T12:13:30.099929Z","shell.execute_reply":"2024-12-16T12:13:30.160432Z"},"jupyter":{"outputs_hidden":false}}
display(df.info())
display(df.describe().T)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# **SUMMARY** <br>
# - All datatypes has declared as the content
# - For convenient column names would replace into lowercase
# - The dataset has a missing values which we'll handle it later
# - Almost in every features indicates a big differences range of values between max value and Q3 Value. Which potentially indicated as *outliers*.
# - Ununiformed scale of feature values, which is normal and we'll handle it in normalization section.
# 
# **STEPS** <br>
# - Replace column names into lowercase
# - Handling Missing values
# - EDA (Identify patterns, correlations, or outliers)
# - Scaling a values

# %% [markdown]
# The data splitted to train and test in the beginning of analysis, so we can focus to do analysis on training data withoud leaking the test set. But the data should be splitted porpotional, so the data on train set and test set splitted by the same distribtuion. Library scikit learn has provided this method named as `StratifiedShuffleSplit`.

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:17:39.073770Z","iopub.execute_input":"2024-12-16T12:17:39.074204Z","iopub.status.idle":"2024-12-16T12:17:39.088565Z","shell.execute_reply.started":"2024-12-16T12:17:39.074171Z","shell.execute_reply":"2024-12-16T12:17:39.087432Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import StratifiedShuffleSplit

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in strat_split.split(df, df['Potability']):
    train_split = df.iloc[train_index]
    test_split = df.iloc[test_index]

train_data = train_split.drop(columns=['Potability'])
train_target = train_split['Potability']

test_data = test_split.drop(columns=['Potability'])
test_target = test_split['Potability']

train_split.shape, test_split.shape, train_target.shape, test_target.shape

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # **Data Cleansing**

# %% [markdown]
# The data we are dealing right now is not clean enough, it might causes the model quality. As we listed above we will do the cleaning process step by step. And right now, we'll make the column name unified as lowercase.
# 
# Other than that. We're using the inheritent from the `TransformerMixin` from scikit-learn library, because we'll the cleaning prosess as a scikit-learn pipeline

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## **1. Lowercasing Column Names**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:21:10.298727Z","iopub.execute_input":"2024-12-16T12:21:10.299258Z","iopub.status.idle":"2024-12-16T12:21:10.313495Z","shell.execute_reply.started":"2024-12-16T12:21:10.299224Z","shell.execute_reply":"2024-12-16T12:21:10.312375Z"},"jupyter":{"outputs_hidden":false}}
class LowerColumnNames(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.columns = X.columns.str.lower()
        return X

train_cleaned = train_data.copy()
print(f"Columns before lowercased:\n \33[32m{df.columns}\33[0m\n")
train_cleaned = LowerColumnNames().fit_transform(train_cleaned)

print(f"Columns after lowercased:\n \33[32m{train_cleaned.columns}\33[0m\n")
train_cleaned.info()

train_split = LowerColumnNames().fit_transform(train_split)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## **Handling Missing Values**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:22:40.788394Z","iopub.execute_input":"2024-12-16T12:22:40.788834Z","iopub.status.idle":"2024-12-16T12:22:40.796471Z","shell.execute_reply.started":"2024-12-16T12:22:40.788796Z","shell.execute_reply":"2024-12-16T12:22:40.795018Z"},"jupyter":{"outputs_hidden":false}}
missing_values = train_cleaned.isna().sum()
print(f'Missing values:\n-------------\n{missing_values}')

# %% [markdown]
# There some columns that indicated misisng values. We'll look deep with visualizaiona as below

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:23:14.140303Z","iopub.execute_input":"2024-12-16T12:23:14.140749Z","iopub.status.idle":"2024-12-16T12:23:14.530204Z","shell.execute_reply.started":"2024-12-16T12:23:14.140711Z","shell.execute_reply":"2024-12-16T12:23:14.528903Z"},"jupyter":{"outputs_hidden":false}}
def plot_missing_values(df):
    missing_values = df.isna().sum()
    missing_values_df = missing_values.reset_index().rename(
        columns={'index':'features', 0:'count_missing'}
    )
    
    rows_with_nan = df[df.isnull().any(axis=1)]
    missing_values_percentage = len(rows_with_nan) / len(df) * 100
    
    display(missing_values_df)
    
    print(f'Number of rows with missing values: {len(rows_with_nan)}')
    print(f'Percentage of all missing values: {missing_values_percentage:.2f}%')
    
    plt.figure(figsize=(15,5))
    sns.barplot(missing_values_df, x='features', y='count_missing', palette='viridis')
    plt.title('Count of missing values each features')

plot_missing_values(train_cleaned)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# **SUMMARY** :<br>
# - Missing values are reaching up to 38% of data, which is high. 
# - We should use the appropriate method for handling. However, dropping it might causes loss a lot of information. Hence, the only way we going to do is the imputation.
# - Due to the dataset is crusial for imputation to be a pricise value (In terms of our dataset contains public health quality). Doing a simple imputation might causes a bias. Hence, we will do the multiple imputation
# 
# **STRATEGY**: <br>
# - Implement MICE (Multiple Imputation by Chained Equations). Ref of explanation from <a href="https://www.machinelearningplus.com/machine-learning/mice-imputation/">Here.</a>
# - Using `IterativeImputer` method from `sklearn` library to apply multiple imputer on missing values. To do that, the prediction values will apply as n-times, which it generates new dataset with different random values each.
# - Use a final dataset to train model.

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:23:20.337772Z","iopub.execute_input":"2024-12-16T12:23:20.338190Z","iopub.status.idle":"2024-12-16T12:23:20.946289Z","shell.execute_reply.started":"2024-12-16T12:23:20.338156Z","shell.execute_reply":"2024-12-16T12:23:20.945148Z"},"jupyter":{"outputs_hidden":false}}
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

class MultipleImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        imp = IterativeImputer(random_state=9122024)
        return pd.DataFrame(imp.fit_transform(X), columns=features)
        
train_cleaned = MultipleImputer().fit_transform(train_cleaned)

plot_missing_values(train_cleaned)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # **EDA**

# %% [markdown]
# As we will create a model based on our data, we will explore and gain the information from our data. So we can understand better how our data is created.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## **1. Distribution**

# %% [markdown]
# As a starter for exploration, let's see the distribution values of each features of our data

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:32:40.284626Z","iopub.execute_input":"2024-12-16T12:32:40.285224Z","iopub.status.idle":"2024-12-16T12:32:40.336234Z","shell.execute_reply.started":"2024-12-16T12:32:40.285170Z","shell.execute_reply":"2024-12-16T12:32:40.335058Z"}}
train_cleaned.describe().T

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:26:13.253358Z","iopub.execute_input":"2024-12-16T12:26:13.253870Z","iopub.status.idle":"2024-12-16T12:26:13.263488Z","shell.execute_reply.started":"2024-12-16T12:26:13.253831Z","shell.execute_reply":"2024-12-16T12:26:13.262147Z"},"jupyter":{"outputs_hidden":false}}
def plot_cols_dist(df, columns, suptitle="Features distribution"):
    """Plot the distribution of the specified columns in the DataFrame."""
    if len(columns) == 1:
        n_cols = 1
        n_rows = 1
    else:
        n_cols = 5
        n_rows = len(columns) // n_cols + (1 if len(columns) % n_cols > 0 else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5), constrained_layout=True)
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    plt.suptitle(suptitle)
    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color="blue", bins=30)
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Remove any unused axes
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:26:34.963755Z","iopub.execute_input":"2024-12-16T12:26:34.964590Z","iopub.status.idle":"2024-12-16T12:26:38.244963Z","shell.execute_reply.started":"2024-12-16T12:26:34.964549Z","shell.execute_reply":"2024-12-16T12:26:38.243797Z"},"jupyter":{"outputs_hidden":false}}
display(plot_cols_dist(train_cleaned, columns=features, suptitle="Train distribution"))
display(train_target.hist())
plt.title('Target distribution')
plt.legend()
plt.show()

# %% [markdown]
# The distribution looks tend to be a normal distribution as an image shaped as a bell curve. and `solids` column looks like a little to be a skewed-right distribution.
# 
# Then if we looked at the second plotting image, the target distribution we can see from that indicated an imbalanced datasets. and to handle that, there is two options on my mind:
# 1. Upsampling for minority data
# 2. Adjusting the threshold of probability so we can adjust the value of the evaluation metrics
# 
# we'll try both

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## **2. Correlation**

# %% [markdown]
# Second we'll identify the correlation between features.

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:34:05.973618Z","iopub.execute_input":"2024-12-16T12:34:05.974689Z","iopub.status.idle":"2024-12-16T12:34:06.440273Z","shell.execute_reply.started":"2024-12-16T12:34:05.974610Z","shell.execute_reply":"2024-12-16T12:34:06.439207Z"},"jupyter":{"outputs_hidden":false}}
def plot_corr(df):
    mat_corr = df.corr()
    plt.figure(figsize=(15,8))
    sns.heatmap(mat_corr, annot=True, fmt='.2f', cmap='coolwarm')

display(plot_corr(train_cleaned))
plt.show()

# %% [markdown]
# There is no so much to see for correlated between columns, but some features like `solid` and `sulfate` indicated as negative correlation between both. as well as `hardness` and `sulfate`, `ph` and `solid`

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## **2. Handling an Outliers**

# %% [markdown]
# This section when we'll handle the outleirs. Before that we'll look deep to the graph. So we can identify the right solution for the problems

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:35:47.852209Z","iopub.execute_input":"2024-12-16T12:35:47.852599Z","iopub.status.idle":"2024-12-16T12:35:48.831171Z","shell.execute_reply.started":"2024-12-16T12:35:47.852567Z","shell.execute_reply":"2024-12-16T12:35:48.830114Z"},"jupyter":{"outputs_hidden":false}}
def plot_cols_boxplot(df, features):
    plt.figure(figsize=(5,8))
    
    ncols = 3
    nrows = len(features) // ncols + (1 if len(features) % ncols > 0 else 0)
    fig, axes = plt.subplots(figsize=(12, 5*nrows), nrows=nrows, ncols=ncols)
    
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(features)-1:  # Check if the index is within bounds, minus 1 for exclude the potability
            sns.boxplot(data=df, x=features[i], ax=ax, palette='viridis')
            ax.set_title(features[i])
        else:
            ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()

plot_cols_boxplot(train_cleaned, features)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:35:48.858795Z","iopub.execute_input":"2024-12-16T12:35:48.859185Z","iopub.status.idle":"2024-12-16T12:35:49.212770Z","shell.execute_reply.started":"2024-12-16T12:35:48.859150Z","shell.execute_reply":"2024-12-16T12:35:49.211554Z"},"jupyter":{"outputs_hidden":false}}
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Dataset with imputed data (replace train_split with your actual DataFrame)
q1 = train_split.quantile(0.25)
q3 = train_split.quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr


# Dictionary to store outliers summary
outliers_summary = {
    'column': [],
    'potability_0': [],
    'potability_1': [],
}

# Calculate outliers for each feature
for col in all_cols:  # Replace 'features' with the list of your feature column names
    outliers = train_split[(train_split[col] < lower_bound[col]) | (train_split[col] > upper_bound[col])]
    outliers_summary['column'].append(col)
    outliers_summary['potability_0'].append(len(outliers[outliers['potability'] == 0]))
    outliers_summary['potability_1'].append(len(outliers[outliers['potability'] == 1]))

# Convert to DataFrame
outliers_summary_df = pd.DataFrame(outliers_summary)

# Melt the DataFrame for better visualization
outliers_summary_melted = outliers_summary_df.melt(id_vars='column', var_name='Potability', value_name='Count')
display(outliers_summary_melted)
# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=outliers_summary_melted, x='column', y='Count', hue='Potability', palette='viridis')
plt.title("Outliers Count for Each Column by Potability")
plt.xticks(rotation=45)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:36:34.388211Z","iopub.execute_input":"2024-12-16T12:36:34.388623Z","iopub.status.idle":"2024-12-16T12:36:34.404611Z","shell.execute_reply.started":"2024-12-16T12:36:34.388583Z","shell.execute_reply":"2024-12-16T12:36:34.403377Z"},"jupyter":{"outputs_hidden":false}}
# Count if the rows has an existing outliers of column
rows_with_outliers = train_split[train_split[(train_split < lower_bound) | (train_split > upper_bound)].any(axis=1)]
potability_0_outliers = rows_with_outliers[rows_with_outliers['potability'] == 0]
potability_1_outliers = rows_with_outliers[rows_with_outliers['potability'] == 1]

outliers_percentage = len(rows_with_outliers) / len(train_split) * 100
potability_0_outliers_percentage = len(potability_0_outliers) / len(train_split) * 100
potability_1_outliers_percentage = len(potability_1_outliers) / len(train_split) * 100

print(f"Number of rows with outliers: \33[33m{len(rows_with_outliers)}\33[0m")
print(f"Number of potability rows with outliers: \33[33m{len(potability_0_outliers)}\33[0m")
print(f"Number of non-potability rows with outliers: \33[33m{len(potability_1_outliers)}\33[0m\n")
print(f"Percentage of rows with outliers: \33[33m{outliers_percentage:.2f}%\33[0m")
print(f"Percentage of potability rows with outliers: \33[33m{potability_0_outliers_percentage:.2f}%\33[0m")
print(f"Percentage of non-potability with outliers: \33[33m{potability_1_outliers_percentage:.2f}%\33[0m")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# **SUMMARY**: <br>
# - The outliers are quite higher as it reaches to 10% of data has been indiceted as an outliers.
# - Handling it with removal or transformation makes it poor quality of data. Which in this case related to data of water quality that requires accurate data.
# 
# **STRATEGY**: <br>
# - Instead of removing or transforming the outliers data, we'll examine it using the robust algorithm like tree-based algorithm `(Decision Tree, Random Forest)`.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # **Preprocessing**
# 
# This section we'll perform the preprocessing process for our model. As we describe before, that all step of preprocessing pipeline will be wrapped with classes inherited from `TransfromerMixin` and `BaseEstimator` so we can put our object into the pipeline method that scikit-learn has provied

# %% [markdown]
# ## **Numerical Cutter Attributes**
# 
# This section we'll cut some numerical features into categorical, hoping it might get an useful new features for developing the model quality. This code below apply the process and we'll try the hands-on, so we can see what happend to the process

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:38:47.884529Z","iopub.execute_input":"2024-12-16T12:38:47.885006Z","iopub.status.idle":"2024-12-16T12:38:47.931719Z","shell.execute_reply.started":"2024-12-16T12:38:47.884966Z","shell.execute_reply":"2024-12-16T12:38:47.930398Z"},"jupyter":{"outputs_hidden":false}}
class NumericalCutterAttribs(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self._columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self._columns:
            col_cut = pd.qcut(X[col], 4, labels=[0,1,2,3])
            df[f"{col}_cut"] = col_cut

        return df

# Hands-on example
cutter_attr = NumericalCutterAttribs(features)
train_cut = cutter_attr.fit_transform(train_cleaned)
train_cut.head()

# %% [markdown]
# The result will add a new feautures exctracted from the numerical features

# %% [markdown]
# Next, we'll put all in one pipeline. Below is the code how to implement it. The last pipeline is scaling all the values. We'll use `StandardScaler` method, it will normalize all values, so all the `mean` will equal to 0 and `standar deviation` equal to 1

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:46:24.754496Z","iopub.execute_input":"2024-12-16T12:46:24.754982Z","iopub.status.idle":"2024-12-16T12:46:24.763905Z","shell.execute_reply.started":"2024-12-16T12:46:24.754942Z","shell.execute_reply":"2024-12-16T12:46:24.762738Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# columns that will cut to the categorical feture
columns_to_cut = ["ph", "hardness"]

# store the pipeline processsing.
preproc_pipe = Pipeline([
    ('columns_lowercase', LowerColumnNames()), # lowercase pipeline
    ('imputer', MultipleImputer()), # imputing the missing values
    ('numerical_cutter', NumericalCutterAttribs(columns_to_cut)), # cut numerical into categorical feature
    ('scaler', StandardScaler()) # scaling
])

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## **Transform Preprocessing**
# 
# Here we'll transform all train and test set with our pipeline

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:48:04.987438Z","iopub.execute_input":"2024-12-16T12:48:04.987912Z","iopub.status.idle":"2024-12-16T12:48:05.087218Z","shell.execute_reply.started":"2024-12-16T12:48:04.987873Z","shell.execute_reply":"2024-12-16T12:48:05.086264Z"},"jupyter":{"outputs_hidden":false}}
X_train_prepared = preproc_pipe.fit_transform(train_data) # fit preproc transformation for X training
X_test_prepared = preproc_pipe.transform(test_data)# fit preproc transformation for X testing

# below just indicate the target values into new variable and reshape to (-1, 1)
y_train = np.array(train_target).reshape(-1, 1)
y_test = np.array(test_target).reshape(-1, 1)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # **Model Selection**

# %% [markdown]
# We'll use 3 different algorithms for our training. it will helps us identify which algorithms are better. we do training with default parameters first. Then we'll se the better result to tune the hyperparameters

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T13:46:04.234879Z","iopub.execute_input":"2024-12-16T13:46:04.235272Z","iopub.status.idle":"2024-12-16T13:46:04.240710Z","shell.execute_reply.started":"2024-12-16T13:46:04.235237Z","shell.execute_reply":"2024-12-16T13:46:04.239589Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## **Model Training**

# %% [markdown]
# Before we jump into training, we'll create the object class and method that we will use many time, so we don't have to write the full codes. This class above just to fit the model using the train and test set that we defined aas initializing constructor. Then we can evaluate using `eval()` method to get the summary of our model. Let's jump in

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:53:16.212573Z","iopub.execute_input":"2024-12-16T12:53:16.213079Z","iopub.status.idle":"2024-12-16T12:53:16.225431Z","shell.execute_reply.started":"2024-12-16T12:53:16.213041Z","shell.execute_reply":"2024-12-16T12:53:16.224214Z"},"jupyter":{"outputs_hidden":false}}
class TestModels():
    def __init__(self, models, section_name='',
                 split_method=None, 
                 X_train=X_train_prepared, 
                 y_train=y_train, 
                 X_test=X_test_prepared,
                 y_test=y_test):

        self._models=models[0] if len(models) == 1 else [model for model in models]
        self._X_train=X_train
        self._y_train=y_train
        self._X_test=X_test
        self._y_test=y_test
        self._scores={
            'train' : [],
            'test' : []
        }
        self._split_method = split_method
        self._section_name=section_name

        if isinstance(self._models, list):
            model_names = [model.__class__.__name__ for model in self._models]
        else:
            model_names = [self._models.__class__.__name__]

        print(f"Models for selection :\n\33[32m{model_names}\33[0m")
        print(f"split method: \33[32m{self._split_method.__class__.__name__}\33[0m")
        
    def model_fit(self):
        model_fitted = []

        # if number of model is higher than 1 to be trained
        if isinstance(self._models, list):
            for model in self._models:
                model_fitted.append(model.fit(self._X_train, self._y_train))
        else:
            model_fitted.append(self._models.fit(self._X_train, self._y_train))
            
        return model_fitted
        
    def eval(self):
        # return list of models to be evaluated
        models = self.model_fit() 

        # Iterate through each models to be trained, 
        # if just one model then the data is trained once with particular model.
        for model in models:

            # train with split method
            if self._split_method != None:
                sm = self._split_method
                cv_scores = cross_val_score(model, self._X_train, self._y_train, cv=sm, scoring='accuracy')
                # average of cross validation scores
                score_train = cv_scores.mean()
                score_test = accuracy_score(self._y_test, model.predict(self._X_test))
                
            # train without split method
            else:
                score_train = accuracy_score(self._y_train, model.predict(self._X_train))
                score_test = accuracy_score(self._y_test, model.predict(self._X_test))

            # store train and test score
            self._scores['train'].append(score_train)
            self._scores['test'].append(score_test)

        # return the dict type for the results
        return {
                'model_names' : [model.__class__.__name__ for model in models], 
                self._section_name+'_train_scores' : self._scores['train'], 
                self._section_name+'_test_scores':self._scores['test']
               }

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T12:53:31.597979Z","iopub.execute_input":"2024-12-16T12:53:31.598374Z","iopub.status.idle":"2024-12-16T12:53:33.428628Z","shell.execute_reply.started":"2024-12-16T12:53:31.598337Z","shell.execute_reply":"2024-12-16T12:53:33.427514Z"},"jupyter":{"outputs_hidden":false}}

models_dict = {
    'lg' : LogisticRegression(),
    'svm' : SVC(),
    'rf' : RandomForestClassifier(random_state=42),
}

models = [model for model in models_dict.values()]

section_name = 'original'
base_models_test = TestModels(models, section_name=section_name)
models_eval = base_models_test.eval()

score_df = pd.DataFrame(models_eval)
score_df.sort_values(by=[f'{section_name}_train_scores', f'{section_name}_test_scores'], ascending=False)

# %% [markdown]
# `RandomForestClassifier` is indicating the better result than other models result. So next we will use it as our main model that we'll develope

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Resampled data

# %% [markdown]
# This as our data is unbalanced, we'll try to resample the minority class and we evaluate how good model can accurately to predict.

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T13:45:42.387914Z","iopub.execute_input":"2024-12-16T13:45:42.388897Z","iopub.status.idle":"2024-12-16T13:45:42.394053Z","shell.execute_reply.started":"2024-12-16T13:45:42.388848Z","shell.execute_reply":"2024-12-16T13:45:42.392961Z"},"jupyter":{"outputs_hidden":false}}
def plot_cluster(source, hue, title=''):
    sns.scatterplot(data=source, x=source.columns[0], y=source.columns[1], hue=hue)
    plt.title(title)
    plt.show()

# %% [markdown]
# We'll see how the data is distributed in lower dimensional level. Hence, first thing we are going todo is to downgrade the dimensional level into just 2 dimenstion, so it can easily plotted into graph. In this case, we will use the pca method.

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T13:47:39.224708Z","iopub.execute_input":"2024-12-16T13:47:39.225101Z","iopub.status.idle":"2024-12-16T13:47:39.636247Z","shell.execute_reply.started":"2024-12-16T13:47:39.225070Z","shell.execute_reply":"2024-12-16T13:47:39.635152Z"},"jupyter":{"outputs_hidden":false}}
_pca = PCA(n_components=2, random_state=42)
X_train_reduced = _pca.fit_transform(X_train_prepared)

train_reduced = np.c_[X_train_reduced, y_train]
train_reduced_df = pd.DataFrame(train_reduced, columns=['pca_1', 'pca_2', 'labels'])

plot_cluster(train_reduced_df, 'labels', title='Training Plot Distribution')

# %% [markdown]
# The plot tells us that the distribution between classes is mixed up. This is quite a problem. Because the model will be more harder in classification. But anyway, we will resample the minority class which is the potable class, so it will equalty distributed with the non potable class. 
# 
# In this caae, we'll take the SMOTE (Synthetic Minority Over-sampling Technique) is a popular technique for dealing with class imbalance in machine learning datasets.

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T13:51:57.373233Z","iopub.execute_input":"2024-12-16T13:51:57.373930Z","iopub.status.idle":"2024-12-16T13:51:57.416634Z","shell.execute_reply.started":"2024-12-16T13:51:57.373889Z","shell.execute_reply":"2024-12-16T13:51:57.415609Z"},"jupyter":{"outputs_hidden":false}}
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

smote = SMOTE(random_state=42)

# resampling training set
train_resampled = smote.fit_resample(X_train_prepared, y_train)
train_resampled = np.c_[train_resampled[1], train_resampled[0]]
train_resampled = shuffle(train_resampled, random_state=42)

X_train_resampled = train_resampled[:, 1:]
y_train_resampled = train_resampled[:, 0]

X_train_resampled[:5], y_train_resampled[:5]

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T13:52:11.157561Z","iopub.execute_input":"2024-12-16T13:52:11.157974Z","iopub.status.idle":"2024-12-16T13:52:11.175727Z","shell.execute_reply.started":"2024-12-16T13:52:11.157939Z","shell.execute_reply":"2024-12-16T13:52:11.174668Z"},"jupyter":{"outputs_hidden":false}}
# Resampling test set
test_resampled = smote.fit_resample(X_test_prepared, y_test)
test_resampled = np.c_[test_resampled[1], test_resampled[0]]
test_resampled = shuffle(test_resampled, random_state=42)

X_test_resampled = test_resampled[:, 1:]
y_test_resampled = test_resampled[:, 0]

X_test_resampled[:5], y_test_resampled[:5]

# %% [markdown]
# Now we'll look the the resampling distribution in lower dimensional space

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T13:52:14.142468Z","iopub.execute_input":"2024-12-16T13:52:14.142861Z","iopub.status.idle":"2024-12-16T13:52:14.556928Z","shell.execute_reply.started":"2024-12-16T13:52:14.142829Z","shell.execute_reply":"2024-12-16T13:52:14.555590Z"},"jupyter":{"outputs_hidden":false}}
X_train_resampled_reduced = _pca.fit_transform(X_train_resampled)
sns.scatterplot(x=X_train_reduced[:,0],y=X_train_reduced[:, 1], label='Original Data')
sns.scatterplot(x=X_train_resampled_reduced[:, 0], y=X_train_resampled_reduced[:, 1], label='Resample Data')

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T13:52:14.763596Z","iopub.execute_input":"2024-12-16T13:52:14.764027Z","iopub.status.idle":"2024-12-16T13:52:15.031494Z","shell.execute_reply.started":"2024-12-16T13:52:14.763993Z","shell.execute_reply":"2024-12-16T13:52:15.030353Z"},"jupyter":{"outputs_hidden":false}}
plt.hist(y_train_resampled)
plt.title('Distribution train data after resampled')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T13:53:23.035017Z","iopub.execute_input":"2024-12-16T13:53:23.035443Z","iopub.status.idle":"2024-12-16T13:53:25.370509Z","shell.execute_reply.started":"2024-12-16T13:53:23.035406Z","shell.execute_reply":"2024-12-16T13:53:25.369272Z"},"jupyter":{"outputs_hidden":false}}
# Retraining on resampled data

section_name='resampled'
models_test_resampled = TestModels(models,
                                   section_name=section_name, 
                                   X_train=X_train_resampled, 
                                   y_train=y_train_resampled, 
                                   X_test=X_test_resampled,
                                   y_test=y_test_resampled
                                  )
models_eval_resampled = models_test_resampled.eval()

score_resampled_df = pd.DataFrame(models_eval_resampled)
score_resampled_df.sort_values(by=[f'{section_name}_train_scores', f'{section_name}_test_scores'], ascending=False)

# %% [markdown]
# The result of model testing after resampling is worst than before resampling, it may causes the mixed class between train and test set is more data mixed. Anyway, we'll stick to the dataset before resampling

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Search Tune

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T13:55:09.204146Z","iopub.execute_input":"2024-12-16T13:55:09.204569Z","iopub.status.idle":"2024-12-16T13:55:09.209823Z","shell.execute_reply.started":"2024-12-16T13:55:09.204534Z","shell.execute_reply":"2024-12-16T13:55:09.208721Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T14:00:22.945052Z","iopub.execute_input":"2024-12-16T14:00:22.945455Z","iopub.status.idle":"2024-12-16T14:06:25.073560Z","shell.execute_reply.started":"2024-12-16T14:00:22.945419Z","shell.execute_reply":"2024-12-16T14:06:25.072007Z"},"jupyter":{"outputs_hidden":false}}

params_rf = {
    'n_estimators' : [100, 150, 250, 850],
    'max_depth' : [25, 35, 85, None],
    'min_samples_split' : [2, 15, 20, 30, 35],
    'min_samples_leaf' : [1, 15, 20]
}

grid_search_rf = GridSearchCV(estimator=models_dict['rf'], 
                                      param_grid=params_rf, 
                                      verbose=1,  
                                      n_jobs=2,
                                      cv=2,
                                      scoring='accuracy', 
                                      return_train_score=True)


grid_search_rf.fit(X_train_prepared, y_train.ravel())

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T14:07:05.986477Z","iopub.execute_input":"2024-12-16T14:07:05.987316Z","iopub.status.idle":"2024-12-16T14:07:05.992977Z","shell.execute_reply.started":"2024-12-16T14:07:05.987271Z","shell.execute_reply":"2024-12-16T14:07:05.991860Z"},"jupyter":{"outputs_hidden":false}}
search_best = grid_search_rf.best_estimator_
print(search_best)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T14:07:13.914815Z","iopub.execute_input":"2024-12-16T14:07:13.915896Z","iopub.status.idle":"2024-12-16T14:07:15.037844Z","shell.execute_reply.started":"2024-12-16T14:07:13.915845Z","shell.execute_reply":"2024-12-16T14:07:15.036659Z"},"jupyter":{"outputs_hidden":false}}
TestModels([search_best], X_train=X_train_resampled, y_train=y_train_resampled, X_test=X_test_prepared).eval()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T14:07:53.670543Z","iopub.execute_input":"2024-12-16T14:07:53.671153Z","iopub.status.idle":"2024-12-16T14:07:54.801733Z","shell.execute_reply.started":"2024-12-16T14:07:53.671074Z","shell.execute_reply":"2024-12-16T14:07:54.800393Z"},"scrolled":true,"jupyter":{"outputs_hidden":false}}
rf_final = RandomForestClassifier(max_depth=35, min_samples_split=15, random_state=42)
TestModels([rf_final], X_train=X_train_resampled, y_train=y_train_resampled, X_test=X_test_prepared).eval()

# %% [markdown]
# The best Estimator after grid search is `RandomForestClassifier(max_depth=35, min_samples_split=15, random_state=42)`, with the train and test score `0.98` and `0.67` respectively. It's inddicating as overfit the model in the training. As we use the randomforest that uses the tree-based model which prone to overfit.
# 
# Anyway, accuracy scores is not the one we can evaluate the model is good or bad. We'll look deep into evaluation to adjust our model to be more matched to the real solution

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # **Evaluation**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T14:08:20.715305Z","iopub.execute_input":"2024-12-16T14:08:20.715728Z","iopub.status.idle":"2024-12-16T14:08:20.720819Z","shell.execute_reply.started":"2024-12-16T14:08:20.715689Z","shell.execute_reply":"2024-12-16T14:08:20.719696Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.model_selection import learning_curve, LearningCurveDisplay
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay

# %% [markdown]
# The code below is to store the final train and test dataset that used to be evaluated

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T14:08:41.631514Z","iopub.execute_input":"2024-12-16T14:08:41.631963Z","iopub.status.idle":"2024-12-16T14:08:41.639601Z","shell.execute_reply.started":"2024-12-16T14:08:41.631922Z","shell.execute_reply":"2024-12-16T14:08:41.638482Z"},"jupyter":{"outputs_hidden":false}}
X_train_final = X_train_prepared
y_train_final = y_train.ravel() # make one dimensional array

X_test_final = X_test_prepared
y_test_final = y_test.ravel()

X_train_final.shape, X_test_final.shape

# %% [markdown]
# We've seen the overfit indicator from scoring the train and test set. Below we look deep down to the visualization of that meaning.

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T14:16:27.167318Z","iopub.execute_input":"2024-12-16T14:16:27.167742Z","iopub.status.idle":"2024-12-16T14:16:27.174435Z","shell.execute_reply.started":"2024-12-16T14:16:27.167700Z","shell.execute_reply":"2024-12-16T14:16:27.173238Z"},"jupyter":{"outputs_hidden":false}}
def plot_learning_curve(train_scores, test_scores):   
    plt.figure()
    plt.plot(train_sizes, train_scores, 'o-', label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', label="Testing score")
    plt.xlabel("Training Sizes")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T14:16:27.630777Z","iopub.execute_input":"2024-12-16T14:16:27.631190Z","iopub.status.idle":"2024-12-16T14:16:42.108984Z","shell.execute_reply.started":"2024-12-16T14:16:27.631154Z","shell.execute_reply":"2024-12-16T14:16:42.107628Z"},"jupyter":{"outputs_hidden":false}}
train_sizes, train_scores, test_scores = learning_curve(
    rf_final,
    X_train_final, 
    y_train_final, 
    verbose=2, 
    shuffle=True, 
    cv=10, 
    n_jobs=2,
    train_sizes=[0.2, 0.33, 0.5, 0.7, 0.9]
)

train_scores_max = np.max(train_scores, axis=1)
test_scores_max = np.max(test_scores, axis=1)

plot_learning_curve(train_scores_max, test_scores_max)

# %% [markdown]
# 

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## **Confusion Matrix**

# %% [markdown]
# The next we see the confusion matrix to get more understanding to our model quality

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T15:27:39.798676Z","iopub.execute_input":"2024-12-16T15:27:39.799501Z","iopub.status.idle":"2024-12-16T15:27:41.027772Z","shell.execute_reply.started":"2024-12-16T15:27:39.799457Z","shell.execute_reply":"2024-12-16T15:27:41.026626Z"},"jupyter":{"outputs_hidden":false}}
model = rf_final
model.fit(X_train_final, y_train_final)

y_test_pred = model.predict(X_test_final)

print(f"test accuracy: {accuracy_score(y_test_final, y_test_pred)}")

print(classification_report(y_test_final, 
                               y_test_pred, 
                               target_names=target_names, 
                               output_dict=False))
display(ConfusionMatrixDisplay.from_estimator(model, X_test_final, y_test_final, cmap='rocket_r'))

# %% [markdown]
# Overall the model can indicate the non-potable and potable quite better. And we can see the total of false positive and True positive are `75` and `38` respectively. In our case, False positive is more risk. Because falsely predicting non-potable water as a potable water is dangerous. In that case, we'll adjust the threshold to match for our requirements.
# 
# To see more clearly we'll visualize the trade off between false and true positve,

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T14:36:39.331388Z","iopub.execute_input":"2024-12-16T14:36:39.331942Z","iopub.status.idle":"2024-12-16T14:36:39.648322Z","shell.execute_reply.started":"2024-12-16T14:36:39.331902Z","shell.execute_reply":"2024-12-16T14:36:39.647064Z"}}
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test_final, y_test_pred)

# %% [markdown]
# AUC = 0.60 means the ability of model randomly guessing the positive value is slightly better than a random guessing, which means the AUC=1.0 is a perfect guessing and AUC=0.5 is random guessing 

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## **Classification Report**

# %% [markdown]
# Below the code for reporting classification metrics from precision, recall, f1-score, and accuracy, and plot the visualization through `confussion matrix` and `precision recall`. 
# 
# Anyway, we should increase the presision values, where the false positive is lowest. Hence, we define threshold uqual to `0.65`

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T15:12:26.275930Z","iopub.execute_input":"2024-12-16T15:12:26.276318Z","iopub.status.idle":"2024-12-16T15:12:26.852897Z","shell.execute_reply.started":"2024-12-16T15:12:26.276287Z","shell.execute_reply":"2024-12-16T15:12:26.851749Z"},"jupyter":{"outputs_hidden":false}}
# Probabilities for the positive class (column 1)
y_test_proba = model.predict_proba(X_test_final)[:, 1]

# adjusting the threshold to 0.7 to be considered as a true class
threshold = 0.7

# Apply the threshold
predictions = np.where(y_test_proba > threshold, 1, 0)
target_names = ['non potable', 'potable']
report_cls = classification_report(y_test_final, 
                               predictions, 
                               target_names=target_names, 
                               output_dict=True)
print(f"th={threshold}\n")


fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
axes = axes.flatten()

# display confusion matrix
display(pd.DataFrame(report_cls).T)
display(ConfusionMatrixDisplay.from_predictions(y_test_final, predictions, cmap='rocket_r', ax=axes[0]))


# precsision recall display
display_prec_rec = PrecisionRecallDisplay.from_predictions(
    y_test_final, predictions, ax=axes[1]
)
_ = display_prec_rec.ax_.set_title("2-class Precision-Recall curve")

baseline_precision = sum(y_test_final) / len(y_test_final)  # Proportion of positives
plt.axhline(y=baseline_precision, color="red", linestyle="--", label="Chance Level")
plt.scatter(x=[report_cls['potable']['recall']],
            y=[report_cls['potable']['precision']])
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)

display(display_prec_rec)

# %% [markdown]
# **Conlusion:** <br>
# The precision is equal to `0.93`. Which is actually good for the model to have an false positive just around 7% of predictions. But as a trade-off the recall is very weak which the value is just `0.054`. means the model falsely predict the potable water as a non potable water around `0.95` of probability.
# 
# However, predicting non potable as a potable water *(False positive)* is more risky than the opposite. Those, from the evaluation we'll adjust the threshold equal to `0.7` to make a precision higher which means decrease the number of false positive.