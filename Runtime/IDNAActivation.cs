using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DNAMatrices;

namespace DNANeuralNetwork
{
    public interface IDNAActivation
    {
        DNAMatrix Activate(DNAMatrix matrix);

        DNAMatrix Derivative(DNAMatrix matrix);

        DNAActivation.ActivationType GetActivationType();

        int GetActivationFunctionIndex();
    }
}

