package com.incra;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

import java.util.Arrays;

/**
 * @author Jeff Risberg
 * @since 05/31/17
 */
public class LinearRegressionExampleApp {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("JavaLinearRegressionExample")
                .master("local[4]")
                .getOrCreate();

        // Load the data.
        Dataset<Row> dataFrame = spark.read().format("libsvm")
                .load("data/mllib/sample_regression_data.txt");
        // columns are "label", "features"

        // Split the data into train and test
        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        LinearRegression lr = new LinearRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        // Fit the model.
        LinearRegressionModel lrModel = lr.fit(train);

        // Print the coefficients and intercept for linear regression.
        System.out.println("Coefficients: " + lrModel.coefficients() +
                " Intercept: " + lrModel.intercept());

        // Summarize the model over the training set and print out some metrics.
        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));

        trainingSummary.residuals().show();
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("r2: " + trainingSummary.r2());

        Dataset<Row> predictions = lrModel.transform(test);
        // Columns are "label", "features", "prediction"

        Arrays.stream(predictions.columns()).forEach(col -> System.out.println(col));

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        Double rmse = evaluator.evaluate(predictions);

        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

        Dataset<Row> predictionsAnd2XLabels = predictions.join(test, "label");
        Arrays.stream(predictionsAnd2XLabels.columns()).forEach(col -> System.out.println(col));

        Dataset<Row> x = predictions.withColumn("foo", functions.lit(1));
        Dataset<Row> y = x.withColumn("foo2", functions.abs(functions.col("label")));
        Dataset<Row> z = y.withColumn("foo3", functions.expr("label + 1"));
        Arrays.stream(z.columns()).forEach(col -> System.out.println(col));
        z.show();

        spark.stop();
    }
}
