namespace IAUN.ML.LogisticRegression;
public class Evaluator
{
    public static EvalResult CrossValidate(List<DatasetInfo> dataset,  int kFolds, double learningRate = 0.1, int epochs = 1000)
    {
        int n = dataset.Count;
        var indeces = Enumerable.Range(0, n).ToArray();
        var rnd = new Random(42);
        indeces = [.. indeces.OrderBy(_ => rnd.Next())];
        int foldSize = n / kFolds;

        double sumAccuracy = 0, sumRecall = 0, sumPrecision = 0, sumF1 = 0;

        for (int k = 0; k < kFolds; k++)
        {

            var trainX = new List<double[]>();
            var trainY = new List<int>();
            var testX = new List<double[]>();
            var testY = new List<int>();

            var start = k * foldSize;
            var end = (k == kFolds - 1) ? n : start + foldSize;


            for (int i = 0; i < n; i++)
            {
                var row = dataset[indeces[i]];
                if (i >= start && i < end)
                {
                    testX.Add(row.Features);
                    testY.Add(row.Label);
                }
                else
                {
                    trainX.Add(row.Features);
                    trainY.Add(row.Label);
                }
            }


            var model = new LogisticRegression(learningRate, epochs);
            model.Train([.. trainX], [.. trainY]);

            int truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;

            for (int i = 0; i < testX.Count; i++)
            {
                int p = model.Predict(testX[i]);
                int a = testY[i];
                if (a == 1 && p == 1) truePositive++;
                if (a == -1 && p == -1) trueNegative++;
                if (a == -1 && p == 1) falsePositive++;
                if (a == 1 && p == -1) falseNegative++;
            }

            var tpfn = truePositive + falseNegative;
            var tpfp = truePositive + falsePositive;

            var accuracy = (double)(truePositive + trueNegative) / (trueNegative + truePositive + falsePositive + falseNegative);
            var precision = tpfp > 0 ? (double)truePositive / tpfp : 0;
            var recall = tpfn > 0 ? (double)truePositive / tpfn : 0;
            var f1 = (precision + recall) > 0 ? (double)(2 * precision * recall) / (precision + recall) : 0;

            sumAccuracy += accuracy;
            sumRecall += recall;
            sumPrecision += precision;
            sumF1 += f1;

        }
        return new EvalResult
        {
            Accuracy = sumAccuracy/kFolds,
            F1Measure = sumF1 / kFolds,
            Precision = sumPrecision / kFolds,
            Recall = sumRecall / kFolds
        };

    }
}
