using UnityEngine;
using System;


namespace DNANeuralNetwork
{
    [Serializable]
    public class NeuralNetworkInfo
    {
        public Vector2Int inputSize;
        public LayerInfo[] layerInfos;
        public Cost.CostType costType;

    }

    [Serializable]
    public class LayerInfo
    {
        //public LayerTypes type;

        // public ActivationLayerInfo activation;
        //public FilterLayerInfo filter;
        public NeuralLayerInfo neural;
        //public PoolingLayerInfo pooling;
    }

    [System.Serializable]
    public struct NeuralLayerInfo
    {
        public int outputSize;
        public Activation.ActivationType activationType;
    }
}

