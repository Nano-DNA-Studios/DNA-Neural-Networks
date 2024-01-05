using MachineLearning.Activations;

namespace MachineLearning
{
	[System.Serializable]
	public class NeuralNetworkSettings
	{
		public int[] networkSize;

		public ActivationType activationType;
		public ActivationType outputActivationType;
		public Cost.CostType costType;

		public double initialLearningRate;
		public double learnRateDecay;
		public int dataPerBatch;
		public double momentum;
		public double regularization;

		public NeuralNetworkSettings()
		{
			activationType = ActivationType.ReLU;
			outputActivationType = ActivationType.Sigmoid;
			costType = Cost.CostType.MeanSquareError;
			initialLearningRate = 0.05;
			learnRateDecay = 0.075;
			dataPerBatch = 32;
			momentum = 0.9;
			regularization = 0.1;
		}
	}
}
