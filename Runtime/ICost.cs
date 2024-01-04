using MachineLearningMath;

namespace MachineLearning
{
	public interface ICost
	{
		double CostFunction(Matrix predictedOutputs, Matrix expectedOutputs);

		Matrix CostDerivative(Matrix predictedOutput, Matrix expectedOutput);

		Cost.CostType CostFunctionType();

		int GetCostIndex();
	}
}

