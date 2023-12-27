using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicDigitRecognition
{
    public class MNISTDataReader
    {
        public static (double[][], double[][]) ReadMNISTData(string imagesFilePath, string labelsFilePath)
        {
            // Read images data
            byte[] imageData = File.ReadAllBytes(imagesFilePath);
            byte[] labelData = File.ReadAllBytes(labelsFilePath);

            int imageSize = 28 * 28; // Size of each image in MNIST dataset
            int numImages = imageData.Length / imageSize;
            int labelOffset = 8; // Offset for labels file header

            double[][] images = new double[numImages][];
            double[] labels = new double[numImages];

            using (BinaryReader imageReader = new BinaryReader(new MemoryStream(imageData)))
            using (BinaryReader labelReader = new BinaryReader(new MemoryStream(labelData)))
            {
                imageReader.ReadInt32(); // Skip magic number
                int numImagesRead = imageReader.ReadInt32();
                imageReader.ReadInt32(); // Skip rows
                imageReader.ReadInt32(); // Skip columns

                labelReader.ReadInt32(); // Skip magic number
                int numLabelsRead = labelReader.ReadInt32();

                if (numImagesRead != numLabelsRead)
                {
                    throw new Exception("Number of images and labels do not match.");
                }

                for (int i = 0; i < numImages; i++)
                {
                    images[i] = new double[imageSize];
                    for (int j = 0; j < imageSize; j++)
                    {
                        images[i][j] = imageReader.ReadByte() / 255.0; // Normalize pixel values to the range [0, 1]
                    }

                    labels[i] = labelReader.ReadByte();
                }
            }

            return (images, OneHotEncode(labels));
        }
        private static double[][] OneHotEncode(double[] labels)
        {
            int numLabels = labels.Length;
            int numClasses = 10; // Digits 0 to 9

            double[][] oneHotEncoded = new double[numLabels][];
            for (int i = 0; i < numLabels; i++)
            {
                oneHotEncoded[i] = new double[numClasses];
                int digit = (int)labels[i];
                oneHotEncoded[i][digit] = 1.0;
            }

            return oneHotEncoded;
        }
    }

}
