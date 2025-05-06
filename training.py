"""
Energy Consumption Model Trainer - Unique Submission Version
Author: Hrishikesh Akkala
Description: Trains a GBTRegressor with cross-validation and saves the best pipeline model.
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("EnergyTrainerCV") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    try:
        # Load training and validation datasets
        train_set = spark.read.csv("dataset/TrainingDataset.csv", header=True, inferSchema=True)
        val_set = spark.read.csv("dataset/ValidationDataset.csv", header=True, inferSchema=True)

        # String indexing for categorical variables
        bldg_indexer = StringIndexer(inputCol="Building Type", outputCol="bldg_index")
        day_indexer = StringIndexer(inputCol="Day of Week", outputCol="day_index")

        # One-hot encoding
        bldg_encoder = OneHotEncoder(inputCol="bldg_index", outputCol="bldg_vec")
        day_encoder = OneHotEncoder(inputCol="day_index", outputCol="day_vec")

        # Feature assembly and scaling
        vectorize = VectorAssembler(
            inputCols=["bldg_vec", "day_vec", "Square Footage", "Number of Occupants",
                       "Appliances Used", "Average Temperature"],
            outputCol="unscaled_features"
        )
        scale_features = StandardScaler(inputCol="unscaled_features", outputCol="features")

        # Define the regressor
        regressor = GBTRegressor(featuresCol="features", labelCol="Energy Consumption")

        # Build pipeline
        stages = [bldg_indexer, day_indexer, bldg_encoder, day_encoder, vectorize, scale_features, regressor]
        training_pipeline = Pipeline(stages=stages)

        # Hyperparameter tuning grid
        grid = ParamGridBuilder() \
            .addGrid(regressor.maxDepth, [3, 5, 7]) \
            .addGrid(regressor.maxIter, [50, 100]) \
            .addGrid(regressor.stepSize, [0.05, 0.1]) \
            .build()

        # Evaluation metric
        metric = RegressionEvaluator(
            labelCol="Energy Consumption", 
            predictionCol="prediction", 
            metricName="rmse"
        )

        # CrossValidator setup
        crossval = CrossValidator(
            estimator=training_pipeline,
            estimatorParamMaps=grid,
            evaluator=metric,
            numFolds=3
        )

        # Train model with CV
        tuned_model = crossval.fit(train_set)

        # Save best model
        tuned_model.bestModel.write().overwrite().save("model/energy_model")

        # Evaluate on validation set
        val_predictions = tuned_model.transform(val_set)
        rmse_score = metric.evaluate(val_predictions)
        print(f"✅ Final RMSE on validation set: {rmse_score:.2f}")

        # Show sample predictions
        val_predictions.select("Energy Consumption", "prediction").show(10)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
