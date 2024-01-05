using MachineLearningMath;

namespace MachineLearning.Cost
{
    public class MeanSquaredError : ICost
    {
        public double CostFunction(Matrix predictedOutputs, Matrix expectedOutputs)
        {
            // cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
            double cost = 0;
            for (int i = 0; i < predictedOutputs.Length; i++)
            {
                double error = predictedOutputs[i] - expectedOutputs[i];
                cost += error * error;
            }
            return 0.5 * cost;
        }

        public Matrix CostDerivative(Matrix predictedOutput, Matrix expectedOutput)
        {
            return predictedOutput - expectedOutput;
        }

        public CostType CostFunctionType()
        {
            return CostType.MeanSquareError;
        }

        public int GetCostIndex()
        {
            return 1;
        }
    }
}
