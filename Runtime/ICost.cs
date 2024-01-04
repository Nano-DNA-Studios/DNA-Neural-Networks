using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MachineLearningMath;

namespace DNANeuralNetwork
{
	public interface IDNACost
	{
		double CostFunction(Matrix predictedOutputs, Matrix expectedOutputs);

		Matrix CostDerivative(Matrix predictedOutput, Matrix expectedOutput);

		DNACost.CostType CostFunctionType();

		int GetCostIndex();
	}
}

