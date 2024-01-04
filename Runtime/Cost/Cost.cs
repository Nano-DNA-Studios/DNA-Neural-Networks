using MachineLearningMath;
using static System.Math;

namespace MachineLearning.Cost
{
    public class Cost
    {
		public static ICost GetCostFromType(CostType type)
		{
			switch (type)
			{
				case CostType.MeanSquareError:
					return new MeanSquaredError();
				case CostType.CrossEntropy:
					return new CrossEntropy();
				default:
					UnityEngine.Debug.LogError("Unhandled cost type");
					return new MeanSquaredError();
			}
		}
	}
}

