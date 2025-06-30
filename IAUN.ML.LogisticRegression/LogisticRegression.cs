namespace IAUN.ML.LogisticRegression;
public class LogisticRegression(double learningRate = 0.1, int epochs = 1000)
{
    private readonly double learningRate = learningRate;
    private readonly int epochs = epochs;
    private double[] w = [];

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public void Train(double[][] X, int[] y)
    {
        var n = X.Length;
        var d = X[0].Length;
        w = new double[d];

        for (int e = 0; e < epochs; e++)
        {
            var gradian = new double[d];
            for (int i = 0; i < n; i++)
            {
                double dot = 0;
                for (int j = 0; j < d; j++)
                {
                    dot += w[j] * X[i][j];
                }

                double yi = y[i];
                double coeff = -yi / (1.0 + Math.Exp(yi * dot));
                for (int j = 0; j < d; j++)
                {
                    gradian[j] += coeff * X[i][j];
                }
            }

            for (int j = 0; j < d; j++)
            {
                w[j] -= learningRate * (gradian[j] / n);
            }
        }

    }

    public int Predict(double[] x)
    {
        double z = 0;
        for (int i = 0; i < w.Length; i++)
        {
            z += w[i] * x[i];
        }
        return Sigmoid(z) >= 0.5 ? 1 : -1;
    }
}
