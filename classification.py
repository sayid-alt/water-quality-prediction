# %% [markdown]
# # **Project Description: Water Quality Prediction for Public Health Protection**
# 
# ## **Goal**
# 
# The primary goal of this project is to protect public health by identifying unsafe water sources to help prevent waterborne diseases. By leveraging machine learning techniques, the project aims to build a predictive model that determines whether water is potable (safe for consumption) or non-potable (unsafe).
# 
# ## **Expected Outcomes**
# 
# - Develop an accurate predictive model that can classify water as potable or non-potable.
# - Gain insights into the most significant factors affecting water potability.
# - Create a model that aids public health initiatives by enabling early identification of unsafe water sources, thereby reducing waterborne illnesses and improving resource allocation.
# 
# >**This project has the potential to contribute meaningfully to public health protection, ensuring safer water supplies for communities worldwide.**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:39:55.853647Z","iopub.execute_input":"2024-12-16T10:39:55.854052Z","iopub.status.idle":"2024-12-16T10:39:55.861045Z","shell.execute_reply.started":"2024-12-16T10:39:55.854018Z","shell.execute_reply":"2024-12-16T10:39:55.859644Z"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import Popen, PIPE
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from IPython.display import display
import os

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # **Data Loading**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:20.801188Z","iopub.execute_input":"2024-12-16T08:30:20.801561Z","iopub.status.idle":"2024-12-16T08:30:20.816149Z","shell.execute_reply.started":"2024-12-16T08:30:20.801521Z","shell.execute_reply":"2024-12-16T08:30:20.814530Z"}}
INPUT_DIR = '/kaggle/input/water-potability/'
WORKING_DIR = '/kaggle/working/'

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:20.818295Z","iopub.execute_input":"2024-12-16T08:30:20.818833Z","iopub.status.idle":"2024-12-16T08:30:20.859721Z","shell.execute_reply.started":"2024-12-16T08:30:20.818773Z","shell.execute_reply":"2024-12-16T08:30:20.858360Z"}}
input_dataset = os.path.join(INPUT_DIR, 'water_potability.csv')

df = pd.read_csv(input_dataset)

df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:20.861052Z","iopub.execute_input":"2024-12-16T08:30:20.861416Z","iopub.status.idle":"2024-12-16T08:30:20.911664Z","shell.execute_reply.started":"2024-12-16T08:30:20.861383Z","shell.execute_reply":"2024-12-16T08:30:20.910441Z"}}
display(df.info())
display(df.describe().T)

# %% [markdown]
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

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:20.914035Z","iopub.execute_input":"2024-12-16T08:30:20.914336Z","iopub.status.idle":"2024-12-16T08:30:20.929959Z","shell.execute_reply.started":"2024-12-16T08:30:20.914307Z","shell.execute_reply":"2024-12-16T08:30:20.928744Z"}}
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

# %% [markdown]
# # **Data Cleansing**

# %% [markdown]
# ## **1. Lowercasing Column Names**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:20.931431Z","iopub.execute_input":"2024-12-16T08:30:20.931956Z","iopub.status.idle":"2024-12-16T08:30:20.948107Z","shell.execute_reply.started":"2024-12-16T08:30:20.931905Z","shell.execute_reply":"2024-12-16T08:30:20.946711Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:20.949593Z","iopub.execute_input":"2024-12-16T08:30:20.950077Z","iopub.status.idle":"2024-12-16T08:30:20.976333Z","shell.execute_reply.started":"2024-12-16T08:30:20.950028Z","shell.execute_reply":"2024-12-16T08:30:20.975059Z"}}
features = train_cleaned.columns.tolist()
all_cols = train_split.columns.tolist()

features, all_cols

# %% [markdown]
# ## **Handling Missing Values**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:20.977853Z","iopub.execute_input":"2024-12-16T08:30:20.978305Z","iopub.status.idle":"2024-12-16T08:30:20.993623Z","shell.execute_reply.started":"2024-12-16T08:30:20.978258Z","shell.execute_reply":"2024-12-16T08:30:20.992442Z"}}
missing_values = train_cleaned.isna().sum()
print(f'Missing values:\n-------------\n{missing_values}')

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:20.995102Z","iopub.execute_input":"2024-12-16T08:30:20.995488Z","iopub.status.idle":"2024-12-16T08:30:21.360142Z","shell.execute_reply.started":"2024-12-16T08:30:20.995444Z","shell.execute_reply":"2024-12-16T08:30:21.358780Z"}}
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

# %% [markdown]
# **SUMMARY** :<br>
# - Missing values are reaching up to 38% of data, which is high. 
# - We should use the appropriate method for handling. However, dropping it might causes loss a lot of information. Hence, the only way we going to do is the imputation.
# - Due to the dataset is crusial for imputation to be a pricise value (In terms of our dataset contains public health quality). Doing a simple imputation might causes a bias. Hence, we will do the multiple imputation
# 
# **STRATEGY**: <br>
# - Implement MICE (Multiple Imputation by Chained Equations). Ref of explanation from <a href="https://www.machinelearningplus.com/machine-learning/mice-imputation/">Here.</a>
# - Using `MICEData` method from `statsmodels` library to apply multiple imputer on missing values. To do that, the prediction values will apply as n-times, which it generates new dataset with different random values each.
# - Use a final dataset to train model.

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:21.362776Z","iopub.execute_input":"2024-12-16T08:30:21.363151Z","iopub.status.idle":"2024-12-16T08:30:21.862191Z","shell.execute_reply.started":"2024-12-16T08:30:21.363117Z","shell.execute_reply":"2024-12-16T08:30:21.860952Z"}}
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

# %% [markdown]
# # **EDA**

# %% [markdown]
# ## **1. Distribution**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:21.865194Z","iopub.execute_input":"2024-12-16T08:30:21.865641Z","iopub.status.idle":"2024-12-16T08:30:21.874058Z","shell.execute_reply.started":"2024-12-16T08:30:21.865603Z","shell.execute_reply":"2024-12-16T08:30:21.872822Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:21.875829Z","iopub.execute_input":"2024-12-16T08:30:21.876277Z","iopub.status.idle":"2024-12-16T08:30:24.902385Z","shell.execute_reply.started":"2024-12-16T08:30:21.876220Z","shell.execute_reply":"2024-12-16T08:30:24.901190Z"}}
display(plot_cols_dist(train_cleaned, columns=features, suptitle="Train distribution"))

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:24.904204Z","iopub.execute_input":"2024-12-16T08:30:24.904662Z","iopub.status.idle":"2024-12-16T08:30:25.193711Z","shell.execute_reply.started":"2024-12-16T08:30:24.904616Z","shell.execute_reply":"2024-12-16T08:30:25.192547Z"}}
display(train_target.hist())
plt.title('Target distribution')
plt.legend()
plt.show()

# %% [markdown]
# ## **2. Correlation**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:25.195068Z","iopub.execute_input":"2024-12-16T08:30:25.195494Z","iopub.status.idle":"2024-12-16T08:30:25.923393Z","shell.execute_reply.started":"2024-12-16T08:30:25.195446Z","shell.execute_reply":"2024-12-16T08:30:25.922319Z"}}
def plot_corr(df):
    mat_corr = df.corr()
    plt.figure(figsize=(15,8))
    sns.heatmap(mat_corr, annot=True, fmt='.2f', cmap='coolwarm')

plot_corr(train_cleaned)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:25.924883Z","iopub.execute_input":"2024-12-16T08:30:25.925312Z","iopub.status.idle":"2024-12-16T08:30:31.554779Z","shell.execute_reply.started":"2024-12-16T08:30:25.925266Z","shell.execute_reply":"2024-12-16T08:30:31.553083Z"}}
from pandas.plotting import scatter_matrix
scatter_matrix(train_cleaned[features], figsize=(15,20))
plt.show()

# %% [markdown]
# ## **2. Handling an Outliers**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:31.556362Z","iopub.execute_input":"2024-12-16T08:30:31.556717Z","iopub.status.idle":"2024-12-16T08:30:32.653211Z","shell.execute_reply.started":"2024-12-16T08:30:31.556683Z","shell.execute_reply":"2024-12-16T08:30:32.652012Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:32.654638Z","iopub.execute_input":"2024-12-16T08:30:32.655027Z","iopub.status.idle":"2024-12-16T08:30:33.025555Z","shell.execute_reply.started":"2024-12-16T08:30:32.654988Z","shell.execute_reply":"2024-12-16T08:30:33.024462Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:33.027107Z","iopub.execute_input":"2024-12-16T08:30:33.027550Z","iopub.status.idle":"2024-12-16T08:30:33.044332Z","shell.execute_reply.started":"2024-12-16T08:30:33.027500Z","shell.execute_reply":"2024-12-16T08:30:33.042993Z"}}
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

# %% [markdown]
# **SUMMARY**: <br>
# - The outliers are quite higher as it reaches to 10% of data has been indiceted as an outliers.
# - Handling it with removal or transformation makes it poor quality of data. Which in this case related to data of water quality that requires accurate data.
# 
# **STRATEGY**: <br>
# - Instead of removing or transforming the outliers data, we'll examine it using the robust algorithm like tree-based algorithm `(Decision Tree, Random Forest)`.

# %% [markdown]
# # **Preprocessing**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:33.045812Z","iopub.execute_input":"2024-12-16T08:30:33.046217Z","iopub.status.idle":"2024-12-16T08:30:33.098164Z","shell.execute_reply.started":"2024-12-16T08:30:33.046164Z","shell.execute_reply":"2024-12-16T08:30:33.096953Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:33.102679Z","iopub.execute_input":"2024-12-16T08:30:33.103062Z","iopub.status.idle":"2024-12-16T08:30:34.389401Z","shell.execute_reply.started":"2024-12-16T08:30:33.103027Z","shell.execute_reply":"2024-12-16T08:30:34.388290Z"}}
plot_corr(train_cut)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T08:30:34.390544Z","iopub.execute_input":"2024-12-16T08:30:34.390848Z","iopub.status.idle":"2024-12-16T08:30:40.372883Z","shell.execute_reply.started":"2024-12-16T08:30:34.390819Z","shell.execute_reply":"2024-12-16T08:30:40.371705Z"}}
scatter_matrix(train_cut, figsize=(15, 20))
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:14.967887Z","iopub.execute_input":"2024-12-16T09:22:14.968335Z","iopub.status.idle":"2024-12-16T09:22:14.974606Z","shell.execute_reply.started":"2024-12-16T09:22:14.968296Z","shell.execute_reply":"2024-12-16T09:22:14.973471Z"}}
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

columns_to_cut = ["ph", "hardness"]

preproc_pipe = Pipeline([
    ('columns_lowercase', LowerColumnNames()),
    ('imputer', MultipleImputer()),
    ('numerical_cutter', NumericalCutterAttribs(columns_to_cut)),
    ('scaler', StandardScaler())
])


# %% [markdown]
# ## **1. Transform Preprocessing**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:14.977258Z","iopub.execute_input":"2024-12-16T09:22:14.977610Z","iopub.status.idle":"2024-12-16T09:22:15.196237Z","shell.execute_reply.started":"2024-12-16T09:22:14.977576Z","shell.execute_reply":"2024-12-16T09:22:15.193901Z"}}
X_train_prepared = preproc_pipe.fit_transform(train_data)
X_test_prepared = preproc_pipe.transform(test_data)

y_train = np.array(train_target).reshape(-1, 1)
y_test = np.array(test_target).reshape(-1, 1)

# %% [markdown]
# # **Model Selection**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:15.202245Z","iopub.execute_input":"2024-12-16T09:22:15.202943Z","iopub.status.idle":"2024-12-16T09:22:15.222142Z","shell.execute_reply.started":"2024-12-16T09:22:15.202876Z","shell.execute_reply":"2024-12-16T09:22:15.220375Z"}}
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, StratifiedShuffleSplit
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

# %% [markdown]
# ## **Model Training**
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:15.225581Z","iopub.execute_input":"2024-12-16T09:22:15.229390Z","iopub.status.idle":"2024-12-16T09:22:15.242198Z","shell.execute_reply.started":"2024-12-16T09:22:15.229310Z","shell.execute_reply":"2024-12-16T09:22:15.241042Z"}}
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
    
        return {
                'model_names' : [model.__class__.__name__ for model in models], 
                self._section_name+'_train_scores' : self._scores['train'], 
                self._section_name+'_test_scores':self._scores['test']
               }

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:15.244095Z","iopub.execute_input":"2024-12-16T09:22:15.244477Z","iopub.status.idle":"2024-12-16T09:22:17.107687Z","shell.execute_reply.started":"2024-12-16T09:22:15.244445Z","shell.execute_reply":"2024-12-16T09:22:17.106655Z"}}

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

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:17.109284Z","iopub.execute_input":"2024-12-16T09:22:17.109728Z","iopub.status.idle":"2024-12-16T09:22:20.669003Z","shell.execute_reply.started":"2024-12-16T09:22:17.109679Z","shell.execute_reply":"2024-12-16T09:22:20.667828Z"}}


# With stratified shuffle split
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)

section_name='sss'
base_models_test = TestModels(models, split_method=sss, section_name='sss') 
score_models_sss = base_models_test.eval()

score_sss_df = pd.DataFrame(score_models_sss)
score_sss_df.sort_values(by=[f'{section_name}_train_scores', f'{section_name}_test_scores'], ascending=False)

# %% [markdown]
# # Resampled data

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:20.670349Z","iopub.execute_input":"2024-12-16T09:22:20.670705Z","iopub.status.idle":"2024-12-16T09:22:20.676293Z","shell.execute_reply.started":"2024-12-16T09:22:20.670673Z","shell.execute_reply":"2024-12-16T09:22:20.675043Z"}}
def plot_cluster(source, hue, title=''):
    sns.scatterplot(data=source, x=source.columns[0], y=source.columns[1], hue=hue)
    plt.title(title)
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:20.677786Z","iopub.execute_input":"2024-12-16T09:22:20.678237Z","iopub.status.idle":"2024-12-16T09:22:21.137910Z","shell.execute_reply.started":"2024-12-16T09:22:20.678190Z","shell.execute_reply":"2024-12-16T09:22:21.136910Z"}}
_pca = PCA(n_components=2, random_state=42)
X_train_reduced = _pca.fit_transform(X_train_prepared)

train_reduced = np.c_[X_train_reduced, y_train]
train_reduced_df = pd.DataFrame(train_reduced, columns=['pca_1', 'pca_2', 'labels'])

plot_cluster(train_reduced_df, 'labels', title='Training Plot Distribution')

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:21.139442Z","iopub.execute_input":"2024-12-16T09:22:21.140255Z","iopub.status.idle":"2024-12-16T09:22:21.183128Z","shell.execute_reply.started":"2024-12-16T09:22:21.140207Z","shell.execute_reply":"2024-12-16T09:22:21.182032Z"}}
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

smote = SMOTE(random_state=42)
train_resampled = smote.fit_resample(X_train_prepared, y_train)
train_resampled = np.c_[train_resampled[1], train_resampled[0]]
train_resampled = shuffle(train_resampled, random_state=42)

X_train_resampled = train_resampled[:, 1:]
y_train_resampled = train_resampled[:, 0]

X_train_resampled[:5], y_train_resampled[:5]

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:21.184257Z","iopub.execute_input":"2024-12-16T09:22:21.184577Z","iopub.status.idle":"2024-12-16T09:22:21.199465Z","shell.execute_reply.started":"2024-12-16T09:22:21.184546Z","shell.execute_reply":"2024-12-16T09:22:21.198334Z"}}
test_resampled = smote.fit_resample(X_test_prepared, y_test)
test_resampled = np.c_[test_resampled[1], test_resampled[0]]
test_resampled = shuffle(test_resampled, random_state=42)

X_test_resampled = test_resampled[:, 1:]
y_test_resampled = test_resampled[:, 0]

X_test_resampled[:5], y_test_resampled[:5]

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:21.202846Z","iopub.execute_input":"2024-12-16T09:22:21.203346Z","iopub.status.idle":"2024-12-16T09:22:21.607818Z","shell.execute_reply.started":"2024-12-16T09:22:21.203310Z","shell.execute_reply":"2024-12-16T09:22:21.606712Z"}}
X_train_resampled_reduced = _pca.fit_transform(X_train_resampled)
sns.scatterplot(x=X_train_reduced[:,0],y=X_train_reduced[:, 1], label='Original Data')
sns.scatterplot(x=X_train_resampled_reduced[:, 0], y=X_train_resampled_reduced[:, 1], label='Resample Data')

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:21.609160Z","iopub.execute_input":"2024-12-16T09:22:21.609483Z","iopub.status.idle":"2024-12-16T09:22:21.809789Z","shell.execute_reply.started":"2024-12-16T09:22:21.609452Z","shell.execute_reply":"2024-12-16T09:22:21.808549Z"}}
plt.hist(y_train_resampled)
plt.title('Distribution train data after resampled')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T09:22:21.811257Z","iopub.execute_input":"2024-12-16T09:22:21.811815Z","iopub.status.idle":"2024-12-16T09:22:24.181489Z","shell.execute_reply.started":"2024-12-16T09:22:21.811776Z","shell.execute_reply":"2024-12-16T09:22:24.180275Z"}}
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
# # Search Tune

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:00:18.062329Z","iopub.execute_input":"2024-12-16T10:00:18.062740Z","iopub.status.idle":"2024-12-16T10:00:18.068294Z","shell.execute_reply.started":"2024-12-16T10:00:18.062700Z","shell.execute_reply":"2024-12-16T10:00:18.067060Z"}}
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:00:40.653327Z","iopub.execute_input":"2024-12-16T10:00:40.653787Z","iopub.status.idle":"2024-12-16T10:14:43.952284Z","shell.execute_reply.started":"2024-12-16T10:00:40.653745Z","shell.execute_reply":"2024-12-16T10:14:43.951186Z"}}

params_rf = {
    'n_estimators' : [100, 150, 250, 850],
    'max_depth' : [25, 35, 85, None],
    'min_samples_split' : [2, 15, 20, 30, 35],
    'min_samples_leaf' : [1, 15, 20]
}

grid_search_rf = GridSearchCV(estimator=models_dict['rf'], 
                                      param_grid=params_rf, 
                                      verbose=3, 
                                      cv=2, 
                                      scoring='accuracy', 
                                      return_train_score=True)


grid_search_rf.fit(X_train_resampled, y_train_resampled)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:28:09.979167Z","iopub.execute_input":"2024-12-16T10:28:09.979596Z","iopub.status.idle":"2024-12-16T10:28:09.987690Z","shell.execute_reply.started":"2024-12-16T10:28:09.979559Z","shell.execute_reply":"2024-12-16T10:28:09.986343Z"}}
print(search_best)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:28:26.227028Z","iopub.execute_input":"2024-12-16T10:28:26.227395Z","iopub.status.idle":"2024-12-16T10:28:36.516930Z","shell.execute_reply.started":"2024-12-16T10:28:26.227365Z","shell.execute_reply":"2024-12-16T10:28:36.515811Z"}}
search_best = grid_search_rf.best_estimator_
TestModels([search_best], X_train=X_train_resampled, y_train=y_train_resampled, X_test=X_test_prepared).eval()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:34:57.364334Z","iopub.execute_input":"2024-12-16T10:34:57.364701Z","iopub.status.idle":"2024-12-16T10:35:07.761235Z","shell.execute_reply.started":"2024-12-16T10:34:57.364669Z","shell.execute_reply":"2024-12-16T10:35:07.760049Z"},"scrolled":true}
rf_final = RandomForestClassifier(n_estimators=850, max_depth=35, random_state=42)
TestModels([rf], X_train=X_train_resampled, y_train=y_train_resampled, X_test=X_test_prepared).eval()

# %% [markdown]
# # **Evaluation**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:29:18.635548Z","iopub.execute_input":"2024-12-16T10:29:18.636001Z","iopub.status.idle":"2024-12-16T10:29:18.642161Z","shell.execute_reply.started":"2024-12-16T10:29:18.635947Z","shell.execute_reply":"2024-12-16T10:29:18.640825Z"}}
from sklearn.model_selection import learning_curve, LearningCurveDisplay
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:29:18.644262Z","iopub.execute_input":"2024-12-16T10:29:18.644619Z","iopub.status.idle":"2024-12-16T10:29:18.662329Z","shell.execute_reply.started":"2024-12-16T10:29:18.644586Z","shell.execute_reply":"2024-12-16T10:29:18.661195Z"}}
y_train_resampled.shape

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:29:18.664065Z","iopub.execute_input":"2024-12-16T10:29:18.664468Z","iopub.status.idle":"2024-12-16T10:29:18.676890Z","shell.execute_reply.started":"2024-12-16T10:29:18.664420Z","shell.execute_reply":"2024-12-16T10:29:18.675808Z"}}
X_train_final = X_train_resampled
y_train_final = y_train_resampled

X_test_final = X_test_prepared
y_test_final = y_test

X_train_final.shape, X_test_final.shape

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:29:18.678303Z","iopub.execute_input":"2024-12-16T10:29:18.678738Z","iopub.status.idle":"2024-12-16T10:29:18.689307Z","shell.execute_reply.started":"2024-12-16T10:29:18.678685Z","shell.execute_reply":"2024-12-16T10:29:18.688224Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:29:55.030819Z","iopub.execute_input":"2024-12-16T10:29:55.031201Z","iopub.status.idle":"2024-12-16T10:34:11.330469Z","shell.execute_reply.started":"2024-12-16T10:29:55.031170Z","shell.execute_reply":"2024-12-16T10:34:11.329275Z"}}
train_sizes, train_scores, test_scores = learning_curve(
    rf_final,
    X_train_final, 
    y_train_final, 
    verbose=5, 
    shuffle=True, 
    cv=10, 
    n_jobs=2,
    train_sizes=[0.2, 0.33, 0.5, 0.7, 0.9]
)

train_scores_max = np.max(train_scores, axis=1)
test_scores_max = np.max(test_scores, axis=1)

plot_learning_curve(train_scores_max, test_scores_max)

# %% [markdown]
# ## **Confusion Matrix**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:34:27.110663Z","iopub.execute_input":"2024-12-16T10:34:27.111917Z","iopub.status.idle":"2024-12-16T10:34:37.362077Z","shell.execute_reply.started":"2024-12-16T10:34:27.111871Z","shell.execute_reply":"2024-12-16T10:34:37.360916Z"}}
model = rf_final
model.fit(X_train_final, y_train_final)

y_test_pred = model.predict(X_test_final)

print(f"test accuracy: {accuracy_score(y_test_final, y_test_pred)}")

ConfusionMatrixDisplay.from_estimator(model, X_test_final, y_test_final, cmap='rocket_r')

# %% [markdown]
# ## **Classification Report**

# %% [code] {"execution":{"iopub.status.busy":"2024-12-16T10:43:47.888887Z","iopub.execute_input":"2024-12-16T10:43:47.889323Z","iopub.status.idle":"2024-12-16T10:43:48.560223Z","shell.execute_reply.started":"2024-12-16T10:43:47.889285Z","shell.execute_reply":"2024-12-16T10:43:48.559105Z"}}
# Probabilities for the positive class (column 1)
y_test_proba = model.predict_proba(X_test_final)[:, 1]

threshold = 0.7

# Apply the threshold
predictions = np.where(y_test_proba > threshold, 1, 0)
target_names = ['non potable', 'potable']
report_cls = classification_report(y_test_final, 
                               predictions, 
                               target_names=target_names, 
                               output_dict=True)
print(f"th={threshold}\n")

# display confusion matrix
display(pd.DataFrame(report_cls).T)
display(ConfusionMatrixDisplay.from_predictions(y_test_final, predictions, cmap='rocket_r'))


# precsision recall display
display_prec_rec = PrecisionRecallDisplay.from_predictions(
    y_test_final, predictions, name="LinearSVC",
)
_ = display_prec_rec.ax_.set_title("2-class Precision-Recall curve")

baseline_precision = sum(y_test_final) / len(y_test_final)  # Proportion of positives
plt.axhline(y=baseline_precision, color="red", linestyle="--", label="Chance Level")
plt.scatter(x=[report_cls['potable']['recall']],
            y=[report_cls['potable']['precision']])
plt.legend()

display(display_prec_rec)

# %% [code]


# %% [code]
