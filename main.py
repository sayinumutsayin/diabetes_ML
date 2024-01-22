import discovering_the_data as dr
import feature_eng as fe
import model as ml
import warnings
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)
warnings.filterwarnings("ignore")
# import seaborn as sns
# from matplotlib import pyplot as plt
# from sklearn.neighbors import LocalOutlierFactor
# import numpy as np

# Step1: Read the data:
df = dr.dataframe_reading("diabetes.csv")

# Step2: Grab the numerical, categorical and "categorical but numerical" columns:
cat_cols, num_cols, cat_but_car = dr.grab_col_names(df)

# Step3: Have a look at the data: Types of columns, Number unique values, missing values
dr.check_data(df)

# Step4: Analysis on numerical columns:
dr.num_summary(df, num_cols, plot=True)

# Step5: Outliers analysis
outlier_cols = []
for col in num_cols:
    print(col, fe.check_outlier(df, col))
    if fe.check_outlier(df, col):
        outlier_cols.append(col)

# Step6: Supress the outliers
for col in outlier_cols:
    fe.replace_with_thresholds(df, col)


# Step7: Analysis between the features and the target
for col in num_cols:
    dr.target_summary(df, "Outcome", col)

# Step6: Correlation analysis: heatmap
dr.correlation_matrix(df,num_cols)

# Step7: Having a look of there are some meaningless values:
df.describe().T

# A human cannot get 0 in his/her glucose level, skin thickness, insulin or BMI. Assigning the mean to these 0 values:
zero_cols = ["Glucose", "SkinThickness", "Insulin", "BMI"]

for col in zero_cols:
    df[col] = df[col].replace(0, df[df[col] != 0][col].mean())

# Step7: Creating new features:

df.loc[(df['Glucose'] < 70), 'GLUCOSE_CAT'] = "hypoglycemia"
df.loc[(df['Glucose'] >= 70) & (df['Glucose'] < 100), 'GLUCOSE_CAT'] = "normal"
df.loc[(df['Glucose'] >= 100) & (df['Glucose'] < 126), 'GLUCOSE_CAT'] = "impaired_glucose"
df.loc[(df['Glucose'] >= 126), 'GLUCOSE_CAT'] = "hyperglycemia"

df.loc[(df['Age'] >= 18) & (df['Age'] < 30), 'AGE_CAT'] = "young_women_"
df.loc[(df['Age'] >= 30) & (df['Age'] < 45), 'AGE_CAT'] = "mature_women"
df.loc[(df['Age'] >= 45) & (df['Age'] < 65), 'AGE_CAT'] = "middle_age"
df.loc[(df['Age'] >= 65) & (df['Age'] < 75), 'AGE_CAT'] = "old"
df.loc[(df['Age'] >= 75) , 'AGE_CAT'] = "elder"

df.loc[(df['BMI'] < 16), 'BMI_CAT'] ="overweak"
df.loc[(df['BMI'] >= 16) & (df['BMI'] < 18.5), 'BMI_CAT'] = "weak"
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25), 'BMI_CAT'] = "normal"
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'BMI_CAT'] = "overweight"
df.loc[(df['BMI'] >= 30) & (df['BMI'] < 70), 'BMI_CAT'] = "obese"

df.loc[(df['BloodPressure'] < 70), 'DIASTOLIC_CAT'] = "low"
df.loc[(df['BloodPressure'] >= 70) & (df['BMI'] < 90), 'DIASTOLIC_CAT'] = "normal"
df.loc[(df['BloodPressure'] >= 90), 'DIASTOLIC_CAT'] = "high"

df.loc[(df['Insulin'] < 120), 'INSULIN_CAT'] = "normal"
df.loc[(df['Insulin'] >= 120), 'INSULIN_CAT'] = "high"

df.loc[(df['Pregnancies'] == 0), 'PREG_CAT'] = "zero_pregnancy"
df.loc[(df['Pregnancies'] > 0) & (df['Pregnancies'] <= 5), 'PREG_CAT'] = "normal"
df.loc[(df['Pregnancies'] > 10), 'PREG_CAT'] = "high_pregnancy"

# Step8: Encoding and scaling:

cat_cols, num_cols, cat_but_car = dr.grab_col_names(df)

cat_cols.remove("Outcome")

df = fe.one_hot_encoder(df,cat_cols)

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    fe.label_encoder(df, col)

non_target_cols = df.drop("Outcome", axis=1).columns.to_list()

df[non_target_cols], scaler = fe.scaling_func(df[non_target_cols], non_target_cols, name="standard")

# Step9: ML models with their performances:

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

performance, models = ml.evaluate_models_new(X,y,plot_imp=True)

