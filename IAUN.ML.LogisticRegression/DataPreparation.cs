
namespace IAUN.ML.LogisticRegression;

public class DataPreparation
{

    public static List<HeartDiseaseInfo> LoadCsv(string path)
    {
        var lines = File.ReadAllLines(path)
                        .Where(l => !string.IsNullOrWhiteSpace(l))
                        .ToList();
        var data = new List<HeartDiseaseInfo>();
        for (int i = 1; i < lines.Count; i++)
        {
            var fields = lines[i].Split(',');
            var o = new HeartDiseaseInfo
            {
                HeartDiseaseId = TryParseInt(fields[0]),
                Age = TryParseInt(fields[1]),
                Sex = fields[2] == "Male",
                Dataset = fields[3],
                CP = fields[4],
                TrestBPS = TryParseInt(fields[5]),
                Cholestrol = TryParseInt(fields[6]),
                FBS = TryParseBool(fields[7].ToLower()),
                Restecg = fields[8],
                Thalch = TryParseInt(fields[9]),
                Exang = TryParseBool(fields[10].ToLower()),
                OldPeak = TryParseDouble(fields[11]),
                Slope = fields[12],
                CA = TryParseInt(fields[13]),
                Thal = fields[14],
                Num = TryParseInt(fields[15]),
            };
            data.Add(o);
        }
        return data;
    }

    public static List<DatasetInfo> ConvertToOneHotEncoding(List<HeartDiseaseInfo> dataset)
    {
        var count = dataset.Count;
        var cpItems = dataset.GroupBy(x => x.CP).Select(x => x.Key).ToList();
        var featuresCount = 6 + cpItems.Count;

        var newDataset = dataset.Select(x => new DatasetInfo { HeartDiseaseId = x.HeartDiseaseId, Features = ColumnToFeature(featuresCount, cpItems, x), Label = x.EncodedNum }).ToList();
        return newDataset;

    }

    private static double[] ColumnToFeature(int featuresCount, List<string> cpItems, HeartDiseaseInfo x)
    {
        var features = new double[featuresCount];
        var index = 0;
        features[index++] = x.Age;
        for (int i = 0; i < cpItems.Count; i++)
        {
            features[index++] = x.CP == cpItems[i] ? 1 : 0;
        }
        features[index++] = x.TrestBPS;
        features[index++] = x.Cholestrol;
        features[index++] = x.Thalch;
        features[index++] = x.OldPeak;
        features[index++] = x.CA;
        return features;
    }

    public static List<DatasetInfo> ConvertToLabelEncoding(List<HeartDiseaseInfo> dataset)
    {
        var count = dataset.Count;
        var cpItems = dataset.GroupBy(x => x.CP).Select(x => x.Key).ToList();
        

        var newDataset = dataset.Select(x => new DatasetInfo { HeartDiseaseId = x.HeartDiseaseId, Features = ColumnToLabel(cpItems, x) ,Label=x.EncodedNum}).ToList();
        return newDataset;
    }

    private static double[] ColumnToLabel( List<string> cpItems, HeartDiseaseInfo x)
    {
        var features = new double[7];
        var index = 0;
        features[index++] = x.Age;
        features[index++] = cpItems.IndexOf(x.CP);
        features[index++] = x.TrestBPS;
        features[index++] = x.Cholestrol;
        features[index++] = x.Thalch;
        features[index++] = x.OldPeak;
        features[index++] = x.CA;
        return features;
    }

    private static double TryParseDouble(string s)
    => double.TryParse(s, out var d) ? (double)d : 0;
    private static int TryParseInt(string s)
        => int.TryParse(s, out var i) ? (int)i : 0;

    private static bool TryParseBool(string s)
        => bool.TryParse(s, out var i) ? (bool)i : false;


}
