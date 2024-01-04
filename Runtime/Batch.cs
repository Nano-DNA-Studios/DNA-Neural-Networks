using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DNANeuralNetwork;

namespace DNANeuralNetwork
{
    public class DNABatch
    {
        public DNADataPoint[] data;
        public int size;

        public DNABatch(int size)
        {
            this.size = size;
            this.data = new DNADataPoint[size];
        }

        public void addData(DNADataPoint data, int index)
        {
            this.data[index] = data;
        }

        public void setData(DNADataPoint[] data)
        {
            this.data = data;
        }
    }
}