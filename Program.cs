using BasicDigitRecognition;

string binDirectoryPath = "C:\\Users\\jleis\\OneDrive - Durham University\\Documents\\Visual Studio 2022\\Projects\\BasicDigitRecognition\\bin\\Debug\\net6.0\\";
string saveFilePath = binDirectoryPath + args[1];
// [COMMAND] create save_path
if (args[0] == "create")
{
    // Example usage
    int inputSize = 28 * 28; // Assuming 28x28 images
    int hiddenSize = 64;
    int outputSize = 10; // Digits 0-9

    NeuralNetwork neuralNetwork = new NeuralNetwork(inputSize, hiddenSize, outputSize);

    neuralNetwork.Save(saveFilePath);
}
// [COMMAND] train save_path epochs
else if (args[0] == "train")
{
    // Training data paths
    string mnistDirectoryPath = binDirectoryPath + "mnist(decompressed)\\";
    string trainImagesFilePath = mnistDirectoryPath + "train-images-idx3-ubyte";
    string trainLabelsFilePath = mnistDirectoryPath + "train-labels-idx1-ubyte";
    // Read training data
    var (trainImages, trainLabels) = MNISTDataReader.ReadMNISTData(trainImagesFilePath, trainLabelsFilePath);
    
   
    // Load the saved neural network
    NeuralNetwork neuralNetwork = NeuralNetwork.Load(saveFilePath);

    // Train the neural network with your dataset (inputs and targets)
    int epochs = int.Parse(args[2]);
    double learningRate = 0.01;

    neuralNetwork.Train(trainImages, trainLabels, epochs, learningRate);

    // Save the trained neural network
    neuralNetwork.Save(saveFilePath);
}
// [COMMAND] test save_path
else if (args[0] == "test")
{
    // Testing data paths
    string mnistDirectoryPath = binDirectoryPath + "mnist(decompressed)\\";
    string testImagesFilePath = mnistDirectoryPath + "t10k-images-idx3-ubyte";
    string testLabelsFilePath = mnistDirectoryPath + "t10k-labels-idx1-ubyte";
    // Read testing data
    var (testImages, testLabels) = MNISTDataReader.ReadMNISTData(testImagesFilePath, testLabelsFilePath);


    // Load the saved neural network
    NeuralNetwork neuralNetwork = NeuralNetwork.Load(saveFilePath);

    // Test neural network's accuracy
    double accuracy = neuralNetwork.Test(testImages, testLabels);
    Console.WriteLine($"Accuracy: {accuracy}");
}
else if (args[0] == "save")
{
    NeuralNetwork neuralNetwork = NeuralNetwork.Load(saveFilePath);
    neuralNetwork.Save(saveFilePath);
}
