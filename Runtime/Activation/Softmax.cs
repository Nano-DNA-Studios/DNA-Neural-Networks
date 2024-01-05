using MachineLearningMath;
using System;
using static System.Math;

namespace MachineLearning.Activations
{
    [Serializable]
    public readonly struct Softmax : IActivation
    {
        public Matrix Activate(Matrix matrix)
        {
            double expSum = 0;

            for (int i = 0; i < matrix.Length; i++)
                expSum += Exp(matrix[i]);

            Matrix activation = new Matrix(matrix.Height, matrix.Width);

            for (int i = 0; i < activation.Length; i++)
                activation[i] = Exp(matrix[i]) / expSum;

            return activation;
        }

        public Matrix Derivative(Matrix matrix)
        {
            double expSum = 0;

            for (int i = 0; i < matrix.Length; i++)
                expSum += Exp(matrix[i]);

            Matrix activation = new Matrix(matrix.Height, matrix.Width);

            for (int i = 0; i < activation.Length; i++)
            {
                double ex = Exp(matrix[i]);
                activation[i] = (ex * expSum - ex * ex) / (expSum * expSum);
            }

            return activation;
        }

        public ActivationType GetActivationType()
        {
            return ActivationType.Softmax;
        }

        public int GetActivationFunctionIndex()
        {
            return 5;
        }
    }
}
