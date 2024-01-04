using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MachineLearningMath;

namespace DNANeuralNetwork
{
    public interface IActivation
    {
        Matrix Activate(Matrix matrix);

        Matrix Derivative(Matrix matrix);

        Activation.ActivationType GetActivationType();

        int GetActivationFunctionIndex();
    }
}

