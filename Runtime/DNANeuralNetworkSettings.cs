using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DNANeuralNetwork
{
	[System.Serializable]
	public class DNANeuralNetworkworkSettings
	{
		public int[] networkSize;

		public DNAActivation.ActivationType activationType;
		public DNAActivation.ActivationType outputActivationType;
		public DNACost.CostType costType;

		public double initialLearningRate;
		public double learnRateDecay;
		public int dataPerBatch;
		public double momentum;
		public double regularization;

		public DNANeuralNetworkworkSettings()
		{
			activationType = DNAActivation.ActivationType.ReLU;
			outputActivationType = DNAActivation.ActivationType.Softmax;
			costType = DNACost.CostType.CrossEntropy;
			initialLearningRate = 0.05;
			learnRateDecay = 0.075;
			dataPerBatch = 32;
			momentum = 0.9;
			regularization = 0.1;
		}
	}
}
