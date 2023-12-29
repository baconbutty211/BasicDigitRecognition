using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace BasicDigitRecognition
{

    [Serializable]
    class NeuralNetwork
    {
        //private int inputSize;
        //private int hiddenSize;
        //private int outputSize;
        //private double[,] weightsInputHidden;
        //private double[,] weightsHiddenOutput;

        //public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        //{
        //    this.inputSize = inputSize;
        //    this.hiddenSize = hiddenSize;
        //    this.outputSize = outputSize;

        //    // Initialize weights with random values
        //    weightsInputHidden = InitializeWeights(inputSize, hiddenSize);
        //    weightsHiddenOutput = InitializeWeights(hiddenSize, outputSize);
        //}

        //private double[,] InitializeWeights(int rows, int cols)
        //{
        //    Random rand = new Random();
        //    double[,] weights = new double[rows, cols];

        //    for (int i = 0; i < rows; i++)
        //    {
        //        for (int j = 0; j < cols; j++)
        //        {
        //            weights[i, j] = rand.NextDouble() * 2 - 1; // Initialize weights between -1 and 1
        //        }
        //    }

        //    return weights;
        //}

        private List<int> layerSizes;
        private List<double[,]> weights;
        private List<double[]> biases;
        public NeuralNetwork(params int[] layerSizes)
        {
            if (layerSizes.Length < 2)
            {
                throw new ArgumentException("Neural network must have at least input and output layers.");
            }

            this.layerSizes = layerSizes.ToList();
            InitializeWeightsAndBiases();
        }

        private void InitializeWeightsAndBiases()
        {
            int numLayers = layerSizes.Count;
            weights = new List<double[,]>();
            biases = new List<double[]>();

            Random random = new Random();
            for (int i = 1; i < numLayers; i++)
            {
                int inputSize = layerSizes[i - 1];
                int outputSize = layerSizes[i];

                double[,] weight = new double[inputSize, outputSize];
                double[] bias = new double[outputSize];

                for (int j = 0; j < outputSize; j++)
                {
                    bias[j] = random.NextDouble() * 2 - 1; // Initialize biases between -1 and 1
                    for (int k = 0; k < inputSize; k++)
                    {
                        weight[k,j] = random.NextDouble() * 2 - 1; // Initialize weights between -1 and 1
                    }
                }

                weights.Add(weight);
                biases.Add(bias);
            }
        }

        private double[] Softmax(double[] x)
        {
            double maxVal = x.Max();
            double[] expX = x.Select(val => Math.Exp(val - maxVal)).ToArray();
            double expSum = expX.Sum();

            return expX.Select(val => val / expSum).ToArray();
        }

        private double[] Sigmoid(double[] x)
        {
            return x.Select(val => 1.0 / (1.0 + Math.Exp(-val))).ToArray();
        }

        private double[] Forward(double[] input)
        {
            double[] currentInput = input;

            for (int i = 0; i < weights.Count; i++)
            {
                double[] layerOutput = VectorMatrixMultiply(currentInput, weights[i]);
                layerOutput = layerOutput.Zip(biases[i], (x, b) => x + b).ToArray(); // Add biases
                currentInput = Sigmoid(layerOutput);
            }

            return currentInput;
        }

        private double[] VectorMatrixMultiply(double[] vector, double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[] result = new double[cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j] += vector[i] * matrix[i, j];
                }
            }

            return result;
        }

        private void Backpropagation(double[] input, double[] target, double learningRate)
        {
            // Feed Forward
            List<double[]> inputs = new List<double[]>();
            List<double[]> outputs = new List<double[]>();
            double[] currentInput = input;
            for (int i = 0; i < weights.Count; i++)
            {
                inputs.Add(currentInput);

                double[] layerOutput = VectorMatrixMultiply(currentInput, weights[i]); // Multiply input by weights
                layerOutput = layerOutput.Zip(biases[i], (x, b) => x + b).ToArray(); // Add biases
                currentInput = (i == weights.Count - 1) ? Softmax(layerOutput) : Sigmoid(layerOutput); // Activate layerOutput

                outputs.Add(currentInput);
            }

            // Calculate output layer error
            List<double[]> errors = new List<double[]>();
            double[] finalOutput = outputs[outputs.Count - 1];
            double[] outputError = new double[finalOutput.Length];
            for (int i = 0; i < finalOutput.Length; i++)
            {
                outputError[i] = finalOutput[i] - target[i];
            }
            errors.Add(outputError);

            // Calculate hidden layer error
            for (int i = outputs.Count - 2; i >= 0; i--)
            {
                double[] hiddenOutput = outputs[i]; // Previous layer errors

                double[] hiddenError = VectorMatrixMultiply(errors[0] /* Previous layer error */, Transpose(weights[i+1]));
                for (int j = 0; j < hiddenOutput.Length; j++)
                {
                    hiddenError[j] *= hiddenOutput[j] * (1 - hiddenOutput[j]);
                }
                errors.Insert(0, hiddenError); // Inserting so that the previous layer of errors is always at index 0 (and remains in layer order).
            }

            // Update weights & biases
            for (int h = 0; h < weights.Count; h++)
            {
                for (int i = 0; i < layerSizes[h+1]; i++)
                {
                    for (int j = 0; j < layerSizes[h]; j++)
                    {
                        weights[h][j, i] -= learningRate * errors[h][i] * inputs[h][j]; // Update weights
                    }
                    biases[h][i] -= learningRate * errors[h][i]; // Update biases
                }
            }
        }

        private double[,] Transpose(double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[,] result = new double[cols, rows];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }

            return result;
        }

        public void Train(double[][] inputs, double[][] targets, int epochs, double learningRate)
        {
            (inputs, targets).Shuffle();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    double[] input = inputs[i];
                    double[] target = targets[i];

                    Backpropagation(input, target, learningRate);
                    if (i % 1000 == 0)
                    {
                        Console.Write($"\r Training epoch {epoch}/{epochs}, image {i}/{inputs.Length}...   ");
                    }
                }
            }
            Console.Write("\n");
        }


        public double Test(double[][] inputs, double[][] targets)
        {
            (inputs, targets).Shuffle();

            int correct = 0;
            int total = inputs.Length;
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] input = inputs[i];
                int prediction = Predict(input); 

                double[] target = targets[i];
                int label = Array.IndexOf(target, target.Max());
                
                if(prediction == label) { correct++; }
            }
            return (double)correct/total;
        }

        public int Predict(double[] input)
        {
            double[] output = Forward(input);
            return Array.IndexOf(output, output.Max());
        }





        public void Save(string filePath)
        {
            using (FileStream stream = new FileStream(filePath, FileMode.Create))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                formatter.Serialize(stream, this);
            }
        }

        /// <summary>
        /// Loads Neural Network from filePath provided.
        /// </summary>
        /// <returns>
        /// Neural Network saved to file.
        /// If no file exists creates new Neural Network
        /// </returns>
        public static NeuralNetwork Load(string filePath)
        {
            using (FileStream stream = new FileStream(filePath, FileMode.Open))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                return (NeuralNetwork)formatter.Deserialize(stream);
            }
        }

        public override string ToString()
        {
            string neuralNetworkArchitecture = "Neural Network Architecture:";
            for (int i = 0; i < layerSizes.Count; i++)
            {
                neuralNetworkArchitecture += $"Layer {i + 1}: {layerSizes[i]} neurons";
            }
            return neuralNetworkArchitecture;
        }
    }
}
