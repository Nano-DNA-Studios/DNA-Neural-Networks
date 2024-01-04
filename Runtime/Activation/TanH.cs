using MachineLearningMath;
using static System.Math;
using System;

namespace MachineLearning.Activations
{
    [Serializable]
    public readonly struct TanH : IActivation
    {
        public Matrix Activate(Matrix matrix)
        {
            Matrix activation = new Matrix(matrix.Height, matrix.Width);

            for (int i = 0; i < activation.Length; i++)
            {
                double e2 = Exp(2 * matrix[i]);
                activation[i] = (e2 - 1) / (e2 + 1);
            }

            return activation;
        }

        public Matrix Derivative(Matrix matrix)
        {
            Matrix activation = new Matrix(matrix.Height, matrix.Width);

            for (int i = 0; i < activation.Length; i++)
            {
                double e2 = Exp(2 * matrix[i]);
                double t = (e2 - 1) / (e2 + 1);
                activation[i] = 1 - t * t;
            }

            return activation;
        }

        public ActivationType GetActivationType()
        {
            return ActivationType.TanH;
        }

        public int GetActivationFunctionIndex()
        {
            return 2;
        }
    }
}
