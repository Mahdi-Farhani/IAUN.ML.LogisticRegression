using IAUN.ML.LogisticRegression;

Console.WriteLine("------ IAUN ML Logistics Regression ----------");
var dataset = DataPreparation.LoadCsv("heart_disease_uci.csv");
Console.WriteLine($"{dataset.Count} items read.");

var oneHotEncodingDataset = DataPreparation.ConvertToOneHotEncoding(dataset);
var labelEncodingDataset = DataPreparation.ConvertToLabelEncoding(dataset);

Console.ReadKey();
