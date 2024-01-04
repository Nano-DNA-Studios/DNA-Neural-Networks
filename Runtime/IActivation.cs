using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MachineLearningMath;

namespace DNANeuralNetwork
{
    public interface IDNAActivation
    {
        Matrix Activate(Matrix matrix);

        Matrix Derivative(Matrix matrix);

        DNAActivation.ActivationType GetActivationType();

        int GetActivationFunctionIndex();
    }
}

