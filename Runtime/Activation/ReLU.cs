using MachineLearningMath;
using static System.Math;
using System;

namespace MachineLearning.Activations
{
    [Serializable]
    public readonly struct ReLU : IActivation
    {
        public Matrix Activate(Matrix matrix)
        {
            Matrix activation = new Matrix(matrix.Height, matrix.Width);

            for (int i = 0; i < activation.Length; i++)
                activation[i] = Max(0, matrix[i]);

            return activation;
        }

        public Matrix Derivative(Matrix matrix)
        {
            Matrix activation = new Matrix(matrix.Height, matrix.Width);

            for (int i = 0; i < activation.Length; i++)
                activation[i] = (matrix[i] > 0) ? 1 : 0;

            return activation;
        }

        public ActivationType GetActivationType()
        {
            return ActivationType.ReLU;
        }

        public int GetActivationFunctionIndex()
        {
            return 3;
        }
    }
}
