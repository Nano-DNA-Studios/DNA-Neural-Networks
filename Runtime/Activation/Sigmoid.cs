using MachineLearningMath;
using static System.Math;
using System;


namespace MachineLearning.Activations
{
    [Serializable]
    public readonly struct Sigmoid : IActivation
    {
        public Matrix Activate(Matrix matrix)
        {
            Matrix activation = new Matrix(matrix.Height, matrix.Width);

            for (int i = 0; i < activation.Length; i++)
                activation[i] = 1.0 / (1 + Exp(-matrix[i]));

            return activation;
        }

        public Matrix Derivative(Matrix matrix)
        {
            Matrix activation = Activate(matrix);

            for (int i = 0; i < activation.Length; i++)
                activation[i] = activation[i] * (1 - activation[i]);

            return activation;
        }

        public ActivationType GetActivationType()
        {
            return ActivationType.Sigmoid;
        }

        public int GetActivationFunctionIndex()
        {
            return 1;
        }
    }
}
