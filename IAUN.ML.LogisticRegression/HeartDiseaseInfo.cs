namespace IAUN.ML.LogisticRegression;
public class HeartDiseaseInfo
{
    public int HeartDiseaseId { get; set; }
    public int Age { get; set; }
    public bool Sex { get; set; }
    public string Dataset { get; set; } = string.Empty;
    public string CP { get; set; } = string.Empty;
    public int TrestBPS { get; set; }
    public int Cholestrol { get; set; }
    public bool FBS { get; set; }
    public string Restecg { get; set; } = string.Empty;
    public int Thalch { get; set; }
    public bool Exang { get; set; }
    public double OldPeak { get; set; }
    public string Slope { get; set; } = string.Empty;
    public int CA { get; set; }
    public string Thal { get; set; } = string.Empty;
    public int Num { get; set; }
    public int EncodedNum => Num == 0 ? -1 : 1;
}