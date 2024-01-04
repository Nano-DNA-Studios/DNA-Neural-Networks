using UnityEngine;
using MachineLearningMath;
using MachineLearning.Cost;

namespace MachineLearning.Parallelization
{
    public class GPUParallelization
    {
        public static ComputeShader LayerOutputGPU;
        public static ComputeShader LayerOutputGPUFloat;
        public static ComputeShader ParallelLayerOutputGPU;
        public static ComputeShader ParallelOutputLayer;
        public static ComputeShader ParallelUpdateGradientsWeights;
        public static ComputeShader ParallelUpdateGradientsBias;
        public static ComputeShader ParallelHiddenLayerNode;

        /// <summary>
        /// Getter and Setter for the Layers Weights
        /// </summary>
        public Matrix weights { get { return Layer.weights; } set { Layer.weights = value; } }

        /// <summary>
        /// Getter and Setter for the Layers Biases
        /// </summary>
        public Matrix biases { get { return Layer.biases; } set { Layer.biases = value; } }

        /// <summary>
        /// Getter and Setter for the Layers Cost Gradient Bias
        /// </summary>
        public Matrix _costGradientBias { get { return Layer.CostGradientBias; } set { Layer.CostGradientBias = value; } }

        /// <summary>
        /// Getter and Setter for the Layers Cost Gradient Weights
        /// </summary>
        public Matrix _costGradientWeight { get { return Layer.CostGradientWeight; } set { Layer.CostGradientWeight = value; } }

        /// <summary>
        /// Reference to the Layer
        /// </summary>
        public Layer Layer { get; set; }

        /// <summary>
        /// Initializes the GPU Parallelization
        /// </summary>
        /// <param name="layer"></param>
        public GPUParallelization(Layer layer)
        {
            Layer = layer;
        }

        /// <summary>
        /// Loads all necessary Compute Shaders
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterAssembliesLoaded)]
        public static void loadAssets()
        {
            //if (SystemInfo.deviceType == DeviceType.Desktop)
            LayerOutputGPU = Resources.Load<ComputeShader>("LayerOutputCalculation");
            LayerOutputGPUFloat = Resources.Load<ComputeShader>("LayerOutputCalculationFloat");
            ParallelLayerOutputGPU = Resources.Load<ComputeShader>("ParrallelLayerOperation");
            ParallelOutputLayer = Resources.Load<ComputeShader>("ParallelOutputLayer");
            ParallelUpdateGradientsWeights = Resources.Load<ComputeShader>("ParallelUpdateGradientsWeights");
            ParallelUpdateGradientsBias = Resources.Load<ComputeShader>("ParallelUpdateGradientsBias");
            ParallelHiddenLayerNode = Resources.Load<ComputeShader>("ParallelHiddenLayerNode");

            if (LayerOutputGPU != null)
            {
                Debug.Log("Loaded!");
            }
            if (ParallelLayerOutputGPU != null)
            {
                Debug.Log("Parallel Loaded!");
            }
            if (ParallelOutputLayer != null)
            {
                Debug.Log("Parallel Output Loaded!");
            }
            if (ParallelUpdateGradientsWeights != null)
            {
                Debug.Log("Parallel Gradients Weights Loaded!");
            }
            if (ParallelUpdateGradientsBias != null)
            {
                Debug.Log("Parallel Gradients Bias Loaded!");
            }
            if (ParallelHiddenLayerNode != null)
            {
                Debug.Log("Parallel Hidden Layer Node");
            }
        }

        public (Matrix weightedInputs, Matrix activation) LayerOutputCalculationTrainingGPUFloat(Matrix inputs)
        {
            Matrix activation = new Matrix(0, 0);
            Matrix weightedInputs = new Matrix(0, 0);
            if (weights.Width == inputs.Height)
            {
                activation = new Matrix(weights.Height, inputs.Width);
                weightedInputs = new Matrix(weights.Height, inputs.Width);

                ComputeShader computeShader = LayerOutputGPUFloat;

                ComputeBuffer weightsVals = new ComputeBuffer(weights.Length, sizeof(float));
                ComputeBuffer biasVals = new ComputeBuffer(biases.Length, sizeof(float));
                ComputeBuffer inputsVals = new ComputeBuffer(inputs.Length, sizeof(float));
                ComputeBuffer activationVals = new ComputeBuffer(activation.Length, sizeof(float));
                ComputeBuffer weightedInputVals = new ComputeBuffer(activation.Length, sizeof(float));
                ComputeBuffer dimensions = new ComputeBuffer(3, sizeof(int) * 2);
                ComputeBuffer activationFunction = new ComputeBuffer(1, sizeof(int));

                //Set Data
                inputsVals.SetData(new MatrixFloat(inputs).Values);
                activationFunction.SetData(new int[] { Layer.activation.GetActivationFunctionIndex() });
                weightsVals.SetData(new MatrixFloat(weights).Values);
                biasVals.SetData(new MatrixFloat(biases).Values);
                dimensions.SetData(new int[] { weights.Width, weights.Height, biases.Width, biases.Height, inputs.Width, inputs.Height });

                //Set Buffers
                computeShader.SetBuffer(0, "weights", weightsVals);
                computeShader.SetBuffer(0, "inputs", inputsVals);
                computeShader.SetBuffer(0, "bias", biasVals);
                computeShader.SetBuffer(0, "dimensions", dimensions);
                computeShader.SetBuffer(0, "weightedInputs", weightedInputVals);
                computeShader.SetBuffer(0, "activation", activationVals);
                computeShader.SetBuffer(0, "activationFunction", activationFunction);

                //Calculate
                computeShader.Dispatch(0, activation.Width, activation.Height, 1);

                //Recieve Results
                MatrixFloat activationMatrix = new MatrixFloat(activation);
                MatrixFloat weightedInputsMatrix = new MatrixFloat(weightedInputs);

                activationVals.GetData(activationMatrix.Values);
                weightedInputVals.GetData(weightedInputsMatrix.Values);

                activation = new Matrix(activationMatrix);
                weightedInputs = new Matrix(weightedInputsMatrix);

                //Clear Memory
                inputsVals.Release();
                activationVals.Release();
                weightedInputVals.Release();
                weightsVals.Release();
                biasVals.Release();
                dimensions.Release();
                activationFunction.Release();
            }
            else
                Debug.Log("Error, Dimensions don't match");

            return (weightedInputs, activation);
        }

        public (Matrix weightedInputs, Matrix activation) LayerOutputCalculationTrainingGPU(Matrix inputs)
        {
            Matrix activation = new Matrix(0, 0);
            Matrix weightedInputs = new Matrix(0, 0);
            if (weights.Width == inputs.Height)
            {
                activation = new Matrix(weights.Height, inputs.Width);
                weightedInputs = new Matrix(weights.Height, inputs.Width);

                ComputeShader computeShader = LayerOutputGPU;

                ComputeBuffer weightsVals = new ComputeBuffer(weights.Length, sizeof(double));
                ComputeBuffer biasVals = new ComputeBuffer(biases.Length, sizeof(double));
                ComputeBuffer inputsVals = new ComputeBuffer(inputs.Length, sizeof(double));
                ComputeBuffer activationVals = new ComputeBuffer(activation.Length, sizeof(double));
                ComputeBuffer weightedInputVals = new ComputeBuffer(activation.Length, sizeof(double));
                ComputeBuffer dimensions = new ComputeBuffer(3, sizeof(int) * 2);
                ComputeBuffer activationFunction = new ComputeBuffer(1, sizeof(int));

                //Set Data
                inputsVals.SetData(inputs.Values);
                activationFunction.SetData(new int[] { Layer.activation.GetActivationFunctionIndex() });
                weightsVals.SetData(weights.Values);
                biasVals.SetData(biases.Values);
                dimensions.SetData(new int[] { weights.Width, weights.Height, biases.Width, biases.Height, inputs.Width, inputs.Height });

                //Set Buffers
                computeShader.SetBuffer(0, "weights", weightsVals);
                computeShader.SetBuffer(0, "inputs", inputsVals);
                computeShader.SetBuffer(0, "bias", biasVals);
                computeShader.SetBuffer(0, "dimensions", dimensions);
                computeShader.SetBuffer(0, "weightedInputs", weightedInputVals);
                computeShader.SetBuffer(0, "activation", activationVals);
                computeShader.SetBuffer(0, "activationFunction", activationFunction);

                //Calculate
                computeShader.Dispatch(0, activation.Width, activation.Height, 1);

                //Receaive Result
                activationVals.GetData(activation.Values);
                weightedInputVals.GetData(weightedInputs.Values);

                //Clear Memory
                inputsVals.Release();
                activationVals.Release();
                weightedInputVals.Release();
                weightsVals.Release();
                biasVals.Release();
                dimensions.Release();
                activationFunction.Release();
            }
            else
                Debug.Log("Error, Dimensions don't match");

            return (weightedInputs, activation);
        }

        public (Tensor weightedInputs, Tensor activation) ParallelLayerOutputCalculationTrainingGPU(Tensor inputs)
        {
            int inputsLength = inputs.Length;
            int outputsLength = Layer.ParallelBatchSize * weights.Height * biases.Width;

            Tensor activation = new Tensor(new int[] { inputs.Dimensions[0], weights.Height, biases.Width });
            Tensor weightedInputs = new Tensor(new int[] { inputs.Dimensions[0], weights.Height, biases.Width });

            ComputeShader computeShader = ParallelLayerOutputGPU;

            //Setup Compute Buffers
            ComputeBuffer dimensions = new ComputeBuffer(3, sizeof(int) * 2);
            ComputeBuffer weightsVals = new ComputeBuffer(weights.Length, sizeof(double));
            ComputeBuffer biasVals = new ComputeBuffer(biases.Length, sizeof(double));
            ComputeBuffer inputsVals = new ComputeBuffer(inputsLength, sizeof(double));
            ComputeBuffer activationVals = new ComputeBuffer(outputsLength, sizeof(double));
            ComputeBuffer weightedInputVals = new ComputeBuffer(outputsLength, sizeof(double));
            ComputeBuffer activationFunction = new ComputeBuffer(1, sizeof(int));

            //Set Data
            weightsVals.SetData(weights.Values);
            biasVals.SetData(biases.Values);
            inputsVals.SetData(inputs.Values);
            dimensions.SetData(new int[] { weights.Width, weights.Height, biases.Width, biases.Height, biases.Width, weights.Width });
            activationFunction.SetData(new int[] { Layer.activation.GetActivationFunctionIndex() });

            //Set Buffers
            computeShader.SetBuffer(0, "weights", weightsVals);
            computeShader.SetBuffer(0, "inputs", inputsVals);
            computeShader.SetBuffer(0, "bias", biasVals);
            computeShader.SetBuffer(0, "dimensions", dimensions);
            computeShader.SetBuffer(0, "weightedInputs", weightedInputVals);
            computeShader.SetBuffer(0, "activation", activationVals);
            computeShader.SetBuffer(0, "activationFunction", activationFunction);

            //Calculate
            computeShader.Dispatch(0, biases.Width, biases.Height, inputs.Dimensions[0]);

            //Receive Result
            activationVals.GetData(activation.Values);
            weightedInputVals.GetData(weightedInputs.Values);

            //Clear Memory
            inputsVals.Release();
            activationVals.Release();
            weightedInputVals.Release();
            dimensions.Release();
            weightsVals.Release();
            biasVals.Release();
            activationFunction.Release();

            return (weightedInputs, activation);
        }

        //Parallel Version
        public void ParallelCalculateOutputLayerNodeValues(ParallelLayerLearnData layerLearnData, Tensor expectedOutput, ICost cost, Matrix expectedOutputDim)
        {
            int expectedOutputLength = expectedOutput.Length;
            int weightedInputLength = layerLearnData.weightedInputs.Length;
            int activationLength = layerLearnData.activations.Length;
            int nodeValuesLength = layerLearnData.nodeValues.Length;

            ComputeShader computeShader = ParallelOutputLayer;

            //Setup Compute Buffers
            ComputeBuffer dimensions = new ComputeBuffer(1, sizeof(int) * 2);

            ComputeBuffer weightedInputs = new ComputeBuffer(weightedInputLength, sizeof(double));

            ComputeBuffer activations = new ComputeBuffer(activationLength, sizeof(double));

            ComputeBuffer expectedOutputs = new ComputeBuffer(expectedOutputLength, sizeof(double));

            ComputeBuffer nodeValues = new ComputeBuffer(nodeValuesLength, sizeof(double));

            ComputeBuffer derivativeTypes = new ComputeBuffer(2, sizeof(int));

            //Set Data
            dimensions.SetData(new int[] { expectedOutputDim.Width, expectedOutputDim.Height });

            weightedInputs.SetData(layerLearnData.weightedInputs.Values);

            activations.SetData(layerLearnData.activations.Values);

            expectedOutputs.SetData(expectedOutput.Values);

            derivativeTypes.SetData(new int[] { cost.GetCostIndex(), Layer.activation.GetActivationFunctionIndex() });

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);
            computeShader.SetBuffer(0, "weightedInputs", weightedInputs);
            computeShader.SetBuffer(0, "activations", activations);
            computeShader.SetBuffer(0, "expectedOutputs", expectedOutputs);
            computeShader.SetBuffer(0, "nodeValues", nodeValues);
            computeShader.SetBuffer(0, "derivativeType", derivativeTypes);

            //Calculate
            computeShader.Dispatch(0, expectedOutputDim.Width, expectedOutputDim.Height, Layer.ParallelBatchSize);

            //Receive Result
            nodeValues.GetData(layerLearnData.nodeValues.Values);

            //Clear Memory
            weightedInputs.Release();
            activations.Release();
            expectedOutputs.Release();
            nodeValues.Release();

            dimensions.Release();
            derivativeTypes.Release();
        }

        public Matrix ParallelUpdateGradientsWeightsGPU(ParallelLayerLearnData layerLearnData)
        {
            int inputsLength = layerLearnData.inputs.Length;
            int nodeValuesLength = layerLearnData.nodeValues.Length;
            int costGradientWeightLength = _costGradientWeight.Length;

            ComputeShader computeShader = ParallelUpdateGradientsWeights;

            //Setup Compute Buffers
            ComputeBuffer dimensions = new ComputeBuffer(3, sizeof(int) * 2);

            ComputeBuffer inputsValues = new ComputeBuffer(inputsLength, sizeof(double));

            ComputeBuffer nodeValuesValues = new ComputeBuffer(nodeValuesLength, sizeof(double));

            ComputeBuffer weightsValues = new ComputeBuffer(costGradientWeightLength, sizeof(double));

            //Set Data
            dimensions.SetData(new int[] { biases.Width, biases.Height, weights.Width, biases.Width, _costGradientWeight.Width, _costGradientWeight.Height });

            inputsValues.SetData(layerLearnData.inputs.Values);

            nodeValuesValues.SetData(layerLearnData.nodeValues.Values);

            weightsValues.SetData(new double[costGradientWeightLength]);

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);
            computeShader.SetBuffer(0, "inputs", inputsValues);
            computeShader.SetBuffer(0, "nodeValues", nodeValuesValues);
            computeShader.SetBuffer(0, "costGradientWeight", weightsValues);

            //Calculate
            computeShader.Dispatch(0, _costGradientWeight.Width, _costGradientWeight.Height, 1); //layerLearnData.Length

            Matrix costGradientWeight = new Matrix(_costGradientWeight.Height, _costGradientWeight.Width);

            //Receive Result
            weightsValues.GetData(costGradientWeight.Values);

            //Clear Memory
            inputsValues.Release();
            nodeValuesValues.Release();
            weightsValues.Release();
            dimensions.Release();

            return costGradientWeight;
        }

        public Matrix ParallelUpdateGradientsBiasGPU(ParallelLayerLearnData layerLearnData)
        {
            int nodeValuesLength = layerLearnData.nodeValues.Length;
            int costGradientBiasLength = _costGradientBias.Length;

            ComputeShader computeShader = ParallelUpdateGradientsBias;

            //Setup Compute Buffers
            ComputeBuffer dimensions = new ComputeBuffer(1, sizeof(int) * 2);

            ComputeBuffer nodeValuesValues = new ComputeBuffer(nodeValuesLength, sizeof(double));

            ComputeBuffer biasValues = new ComputeBuffer(costGradientBiasLength, sizeof(double));

            //Set Data
            dimensions.SetData(new int[] { biases.Width, biases.Height });

            nodeValuesValues.SetData(layerLearnData.nodeValues.Values);

            biasValues.SetData(new double[costGradientBiasLength]);

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);

            computeShader.SetBuffer(0, "nodeValues", nodeValuesValues);
            computeShader.SetBuffer(0, "costGradientBias", biasValues);

            //Calculate
            computeShader.Dispatch(0, _costGradientBias.Width, _costGradientBias.Height, 1); //layerLearnData.Length

            Matrix costGradientBias = new Matrix(_costGradientBias.Height, _costGradientBias.Width);

            //Receive Result
            biasValues.GetData(costGradientBias.Values);

            //Clear Memory
            nodeValuesValues.Release();
            biasValues.Release();
            dimensions.Release();

            return costGradientBias;
        }

        public void ParallelHiddenLayerNodeCalc(ParallelLayerLearnData layerLearnData, Layer oldLayer, Tensor oldNodeValues)
        {
            int layerLearnDataLength = layerLearnData.weightedInputs.Length;
            int oldNodeValuesLength = oldNodeValues.Length;
            int oldLayerWeightsLength = oldLayer.weights.Length;
            int nodeValuesLength = layerLearnData.nodeValues.Length;

            ComputeShader computeShader = ParallelHiddenLayerNode;

            ComputeBuffer dimensions = new ComputeBuffer(4, sizeof(int) * 2);
            ComputeBuffer oldLayerWeights = new ComputeBuffer(oldLayerWeightsLength, sizeof(double));
            ComputeBuffer oldNodeVals = new ComputeBuffer(oldNodeValuesLength, sizeof(double));
            ComputeBuffer weightedInputs = new ComputeBuffer(layerLearnDataLength, sizeof(double));
            ComputeBuffer nodeValues = new ComputeBuffer(nodeValuesLength, sizeof(double));
            ComputeBuffer activationDerivative = new ComputeBuffer(1, sizeof(int));

            //Set Data 
            dimensions.SetData(new int[] { oldLayer.weights.Height, oldLayer.weights.Width, oldLayer.biases.Width, oldLayer.biases.Height, biases.Width, biases.Height, biases.Width, biases.Height });
            oldLayerWeights.SetData(oldLayer.weights.Values); //Transpose
            oldNodeVals.SetData(oldNodeValues.Values);
            weightedInputs.SetData(layerLearnData.weightedInputs.Values);
            nodeValues.SetData(new double[nodeValuesLength]);
            activationDerivative.SetData(new int[] { Layer.activation.GetActivationFunctionIndex() });

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);

            computeShader.SetBuffer(0, "weightedInputs", weightedInputs);
            computeShader.SetBuffer(0, "oldNodeValues", oldNodeVals);
            computeShader.SetBuffer(0, "oldLayerWeights", oldLayerWeights);
            computeShader.SetBuffer(0, "nodeValues", nodeValues);
            computeShader.SetBuffer(0, "activationDerivative", activationDerivative);

            //Calculate
            computeShader.Dispatch(0, biases.Width, biases.Height, Layer.ParallelBatchSize);

            //Receive Result
            nodeValues.GetData(layerLearnData.nodeValues.Values);

            //Clear Memory
            weightedInputs.Release();
            activationDerivative.Release();
            oldLayerWeights.Release();
            oldNodeVals.Release();
            nodeValues.Release();
            dimensions.Release();
        }
    }
}

