using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DNANeuralNetwork;


namespace DNANeuralNetwork
{
    [System.Serializable]
    public class NeuralNetworkInfo
    {
        public Vector2Int inputSize;
        public LayerInfo[] layerInfos;
        public DNACost.CostType costType;

    }

    [System.Serializable]
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
        public DNAActivation.ActivationType activationType;
    }
}

