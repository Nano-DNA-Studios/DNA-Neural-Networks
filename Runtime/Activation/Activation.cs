using System;

namespace MachineLearning.Activations
{
	[Serializable]
	public class Activation
	{
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
	}
}