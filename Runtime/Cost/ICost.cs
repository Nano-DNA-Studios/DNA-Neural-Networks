using MachineLearningMath;

namespace MachineLearning.Cost
{
	public interface ICost
	{
		double CostFunction(Matrix predictedOutputs, Matrix expectedOutputs);

		Matrix CostDerivative(Matrix predictedOutput, Matrix expectedOutput);

		CostType CostFunctionType();

		int GetCostIndex();
	}
}

