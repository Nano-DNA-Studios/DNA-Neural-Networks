using MachineLearningMath;
using static System.Math;
using System;

namespace DNANeuralNetwork
{
	[Serializable]
	public struct Activation
	{
		[Serializable]
		public enum ActivationType
		{
			Sigmoid,
			TanH,
			ReLU,
			SiLU,
			Softmax
		}

		public static IActivation GetActivationFromType(ActivationType type)
		{
			switch (type)
			{
				case ActivationType.Sigmoid:
					return new Sigmoid();
				case ActivationType.TanH:
					return new TanH();
				case ActivationType.ReLU:
					return new ReLU();
				case ActivationType.SiLU:
					return new SiLU();
				case ActivationType.Softmax:
					return new Softmax();
				default:
					UnityEngine.Debug.LogError("Unhandled activation type");
					return new Sigmoid();
			}
		}

		public static IActivation GetActivationFromIndex(int activationIndex)
		{
			switch (activationIndex)
			{
				case 1:
					return new Sigmoid();
				case 2:
					return new TanH();
				case 3:
					return new ReLU();
				case 4:
					return new SiLU();
				case 5:
					return new Softmax();
				default:
					UnityEngine.Debug.LogError("Unhandled activation type");
					return new Sigmoid();
			}
		}

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

			public int GetActivationFunctionIndex ()
            {
				return 1;
            }
		}

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
}