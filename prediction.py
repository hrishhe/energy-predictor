import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

def main(test_data_path):
    # Initialize Spark session
    spark = SparkSession.builder.appName("EnergyPredictorSubmission").getOrCreate()

    try:
        # Load validation data
        test_data = spark.read.csv(test_data_path, header=True, inferSchema=True)

        # Load trained pipeline model
        model = PipelineModel.load("model/energy_model")

        # Generate predictions
        predicted_data = model.transform(test_data)

        # Evaluate model using RMSE
        evaluator = RegressionEvaluator(
            labelCol="Energy Consumption",
            predictionCol="prediction",
            metricName="rmse"
        )
        rmse_score = evaluator.evaluate(predicted_data)

        print(f"RMSE on test data: {rmse_score:.2f}")

        # Display a sample of predictions
        predicted_data.select("Energy Consumption", "prediction").show(10)

    finally:
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("⚠️  Usage: python prediction.py <path_to_test_dataset>")
        sys.exit(1)

    main(sys.argv[1])