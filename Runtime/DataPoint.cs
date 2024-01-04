using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DNANeuralNetwork;
using MachineLearningMath;

namespace DNANeuralNetwork
{
    public class DataPoint
    {
        public Matrix inputs;
        public Matrix expectedOutputs;
        public int label;

        public DataPoint(Matrix inputs, int label, int numLabels)
        {
            this.inputs = inputs;
            this.label = label;
            expectedOutputs = CreateOneHot(label, numLabels);
        }

        public static Matrix CreateOneHot(int index, int num)
        {
            Matrix expOut = new Matrix(num, 1);
            expOut[index, 0] = 1;
            return expOut;
        }

        public DataPoint(Matrix inputs, Matrix outputs)
        {
            this.inputs = inputs;
            this.expectedOutputs = outputs;
            this.label = 0;
        }
    }

}
