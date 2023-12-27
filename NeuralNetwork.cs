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
        private int inputSize;
        private int hiddenSize;
        private int outputSize;
        private double[,] weightsInputHidden;
        private double[,] weightsHiddenOutput;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            // Initialize weights with random values
            weightsInputHidden = InitializeWeights(inputSize, hiddenSize);
            weightsHiddenOutput = InitializeWeights(hiddenSize, outputSize);
        }

        private double[,] InitializeWeights(int rows, int cols)
        {
            Random rand = new Random();
            double[,] weights = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    weights[i, j] = rand.NextDouble() * 2 - 1; // Initialize weights between -1 and 1
                }
            }

            return weights;
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
            // Calculate input to hidden layer
            double[] hiddenInput = VectorMatrixMultiply(input, weightsInputHidden);
            double[] hiddenOutput = Sigmoid(hiddenInput);

            // Calculate hidden to output layer
            double[] finalInput = VectorMatrixMultiply(hiddenOutput, weightsHiddenOutput);
            double[] finalOutput = Softmax(finalInput);

            return finalOutput;
        }

        private double[] VectorMatrixMultiply(double[] vector, double[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[] result = new double[cols];

            for (int j = 0; j < cols; j++)
            {
                for (int i = 0; i < rows; i++)
                {
                    result[j] += vector[i] * matrix[i, j];
                }
            }

            return result;
        }

        private void Backpropagation(double[] input, double[] target, double learningRate)
        {
            // Forward pass
            double[] hiddenInput = VectorMatrixMultiply(input, weightsInputHidden);
            double[] hiddenOutput = Sigmoid(hiddenInput);
            double[] finalInput = VectorMatrixMultiply(hiddenOutput, weightsHiddenOutput);
            double[] finalOutput = Softmax(finalInput);

            // Calculate output layer error
            double[] outputError = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                outputError[i] = finalOutput[i] - target[i];
            }

            // Calculate hidden layer error
            double[] hiddenError = VectorMatrixMultiply(outputError, Transpose(weightsHiddenOutput));
            for (int i = 0; i < hiddenSize; i++)
            {
                hiddenError[i] *= hiddenOutput[i] * (1 - hiddenOutput[i]);
            }

            // Update weights
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    weightsInputHidden[i, j] -= learningRate * hiddenError[j] * input[i];
                }
            }

            for (int i = 0; i < hiddenSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    weightsHiddenOutput[i, j] -= learningRate * outputError[j] * hiddenOutput[i];
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
    }
}
