# JTXBigData

# Electricity Load Prediction

## Overview
This project implements a machine learning solution for predicting electricity demand using PySpark. Our model significantly outperforms existing approaches with an RMSE of 79.18, compared to a benchmark LSTM model's RMSE of 182.

## Features
- **Advanced Time Series Analysis**: Uses lagged features and moving averages to capture temporal patterns
- **Multi-feature Model**: Incorporates weather data and temporal variables (day of week, hour, holidays)
- **High Accuracy**: Achieves state-of-the-art prediction accuracy for electricity load forecasting
- **Scalable Architecture**: Built on Apache Spark for handling large-scale data processing

## Prerequisites
```
pip install pyspark findspark
pip install pandas openpyxl
pip install matplotlib
pip install numpy seaborn
pip install pymongo dnspython
```

## Data Processing Pipeline
The project implements a comprehensive data processing pipeline that includes:

1. **Feature Engineering**:
   - Lagged demand features: `week_X-2`, `week_X-3`, `week_X-4`
   - Moving averages: `MA_X-4`
   - Temporal features: `dayOfWeek`, `weekend`, `holiday`, `Holiday_ID`, `hourOfDay`
   - Weather data: `T2M_toc` (temperature)

2. **Data Preprocessing**:
   - Feature assembly using `VectorAssembler`
   - Feature scaling with `StandardScaler`

3. **Model Training**:
   - Random Forest Regressor implementation
   - Hyperparameter optimization

## Model Performance
- **RMSE**: 79.18
- Significantly outperforms the benchmark LSTM model (RMSE: 182)
- Excellent prediction accuracy with demand values ranging from 900 to 1300

## Feature Importance
Our analysis identified the most significant features influencing electricity demand:

![Feature Importance Chart]

The model leverages these feature importance insights to make highly accurate predictions.

## Usage
```python
# Initialize Spark session
spark = SparkSession.builder \
    .appName("Electricity Load Prediction") \
    .getOrCreate()

# Load data
pandas_df = pd.read_excel("./data/train_dataframes.xlsx")
spark_df = spark.createDataFrame(pandas_df)

# Define features and create pipeline
feature_columns = ["week_X-2", "week_X-3", "week_X-4", "MA_X-4", 
                  "dayOfWeek", "weekend", "holiday", "Holiday_ID", 
                  "hourOfDay", "T2M_toc"]
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="assembled_features")
scaler = StandardScaler(inputCol="assembled_features", outputCol="features")
rf = RandomForestRegressor(featuresCol="features", labelCol="DEMAND")
pipeline = Pipeline(stages=[vector_assembler, scaler, rf])

# Train model
model = pipeline.fit(spark_df)

# Make predictions
predictions = model.transform(new_data)
```

## Evaluation
The model evaluation demonstrates exceptional performance:

```python
# Evaluate model
evaluator = RegressionEvaluator(labelCol="DEMAND", predictionCol="prediction")
rmse = evaluator.evaluate(transformed_test, {evaluator.metricName: "rmse"})
print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
# Output: Root Mean Squared Error (RMSE) on test data = 79.18153209985655
```

## Visualization
The project includes visualization tools to assess model performance:

```python
# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(actual_values, predicted_values, alpha=0.5)
plt.plot([x_min, x_max], [x_min, x_max], 'r--', lw=2)
plt.title('Actual vs. Predicted Electricity Demand')
plt.xlabel('Actual DEMAND')
plt.ylabel('Predicted DEMAND')
plt.grid(True)
plt.show()
```

```
