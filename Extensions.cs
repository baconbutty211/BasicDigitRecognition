using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace BasicDigitRecognition
{
    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);
            else
                Console.WriteLine("Bit converter is big endian");
            return BitConverter.ToInt32(bytes, 0);
        }

        private static Random random = new Random();
        public static void Shuffle<T>(this T[] array)
        {
            int n = array.Length;
            for (int i = n - 1; i > 0; i--)
            {
                int j = random.Next(0, i + 1);
                Swap(array, i, j);
            }
        }
        private static void Swap<T>(T[] array, int i, int j)
        {
            T temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }

        /// <summary>
        /// Shuffles 2 input arrays in the same way. (To preserve their relation: image[random] matches with label[random]).
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="input"></param>
        public static void Shuffle<T>(this (T[], T[]) input)
        {
            int n = input.Item1.Length;
            for (int i = n - 1; i > 0; i--)
            {
                int j = random.Next(0, i + 1);
                Swap(input.Item1, i, j);
                Swap(input.Item2, i, j);
            }
        }
    }
}
