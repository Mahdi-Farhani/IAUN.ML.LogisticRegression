using IAUN.ML.LogisticRegression;

Console.WriteLine("------ IAUN ML Logistics Regression ----------");
var dataset = DataPreparation.LoadCsv("heart_disease_uci.csv");
Console.WriteLine($"{dataset.Count} items read.");

var oneHotEncodingDataset = DataPreparation.ConvertToOneHotEncoding(dataset);
var labelEncodingDataset = DataPreparation.ConvertToLabelEncoding(dataset);


Console.WriteLine("--------- ONE HOT Encoding ----------");
var oneHotEncodingResult = Evaluator.CrossValidate(oneHotEncodingDataset, 10);
Console.WriteLine($"Accuracy  :\t{oneHotEncodingResult.Accuracy:F4}");
Console.WriteLine($"Precision :\t{oneHotEncodingResult.Precision:F4}");
Console.WriteLine($"Recall    :\t{oneHotEncodingResult.Recall:F4}");
Console.WriteLine($"F1Measure :\t{oneHotEncodingResult.F1Measure:F4}");


Console.WriteLine("---------  Label Encoding  ----------");
var labelEncodingResult = Evaluator.CrossValidate(labelEncodingDataset, 10);
Console.WriteLine($"Accuracy  :\t{labelEncodingResult.Accuracy:F4}");
Console.WriteLine($"Precision :\t{labelEncodingResult.Precision:F4}");
Console.WriteLine($"Recall    :\t{labelEncodingResult.Recall:F4}");
Console.WriteLine($"F1Measure :\t{labelEncodingResult.F1Measure:F4}");



Console.ReadKey();
