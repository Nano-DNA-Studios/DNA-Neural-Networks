using MachineLearningMath;

namespace DNANeuralNetwork
{
	public interface ICost
	{
		double CostFunction(Matrix predictedOutputs, Matrix expectedOutputs);

		Matrix CostDerivative(Matrix predictedOutput, Matrix expectedOutput);

		Cost.CostType CostFunctionType();

		int GetCostIndex();
	}
}

