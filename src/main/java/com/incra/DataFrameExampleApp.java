package com.incra;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.*;

/**
 * @author Jeff Risberg
 * @since 05/14/17
 */
public class DataFrameExampleApp {
    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .appName("JavaDataFrameExample")
                .master("local[4]")
                .getOrCreate();

        // Create an RDD of Person objects and register it as a table.
        Dataset<Row> df = spark.read().option("header", "true").csv("./data/peopleWithHeader.txt");
        df.show();

        // This is relatively fast
        Dataset<Row> dfs = spark.read().json("./data/employee.json");
        dfs.show();
        dfs.printSchema();

        // this is faster
        dfs.filter(dfs.col("age").gt(23)).show();

        // This seems to take quite a while, perhaps because GroupBy and Count are
        // creating a new RDD, and count requires a collection oer hte clusters.
        dfs.groupBy("age").count().show();

        // Create an RDD of Person objects from a text file
        JavaRDD<Person> peopleRDD = spark.read()
                .textFile("./data/people.txt")
                .javaRDD()
                .map(new Function<String, Person>() {
                    @Override
                    public Person call(String line) throws Exception {
                        String[] parts = line.split(",");
                        Person person = new Person();
                        person.setName(parts[0]);
                        person.setAge(Integer.parseInt(parts[1].trim()));
                        return person;
                    }
                });

        // Apply a schema to an RDD of JavaBeans to get a DataFrame
        Dataset<Row> peopleDF = spark.createDataFrame(peopleRDD, Person.class);

        // Register the DataFrame as a temporary view
        peopleDF.createOrReplaceTempView("people");

        // SQL statements can be run by using the sql methods provided by spark
        Dataset<Row> teenagersDF = spark.sql("SELECT name FROM people WHERE age BETWEEN 13 AND 19");

        // The columns of a row in the result can be accessed by field index
        Encoder<String> stringEncoder = Encoders.STRING();
        Dataset<String> teenagerNamesByIndexDF = teenagersDF.map(new MapFunction<Row, String>() {
            @Override
            public String call(Row row) throws Exception {
                return "Name: " + row.getString(0);
            }
        }, stringEncoder);
        teenagerNamesByIndexDF.show();

        // or accessed by field name
        Dataset<String> teenagerNamesByFieldDF = teenagersDF.map(new MapFunction<Row, String>() {
            @Override
            public String call(Row row) throws Exception {
                return "Name: " + row.<String>getAs("name");
            }
        }, stringEncoder);
        teenagerNamesByFieldDF.show();

        // Summary statistics are also available.
        Dataset<Row> summary = dfs.describe();
        summary.show();

        spark.stop();
    }
}
