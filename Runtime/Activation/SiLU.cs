using MachineLearningMath;
using static System.Math;
using System;

namespace MachineLearning.Activations
{
    [Serializable]
    public readonly struct SiLU : IActivation
    {
        public Matrix Activate(Matrix matrix)
        {
            Matrix activation = new Matrix(matrix.Height, matrix.Width);

            for (int i = 0; i < activation.Length; i++)
                activation[i] = matrix[i] / (1 + Exp(-matrix[i]));

            return activation;
        }

        public Matrix Derivative(Matrix matrix)
        {
            Matrix activation = new Matrix(matrix.Height, matrix.Width);

            for (int i = 0; i < activation.Length; i++)
            {
                double sig = 1 / (1 + Exp(-matrix[i]));
                activation[i] = matrix[i] * sig * (1 - sig) + sig;
            }

            return activation;
        }

        public ActivationType GetActivationType()
        {
            return ActivationType.SiLU;
        }

        public int GetActivationFunctionIndex()
        {
            return 4;
        }
    }
}
