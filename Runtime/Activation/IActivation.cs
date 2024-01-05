using MachineLearningMath;

namespace MachineLearning.Activations
{
    public interface IActivation
    {
        Matrix Activate(Matrix matrix);

        Matrix Derivative(Matrix matrix);

        ActivationType GetActivationType();

        int GetActivationFunctionIndex();
    }
}

