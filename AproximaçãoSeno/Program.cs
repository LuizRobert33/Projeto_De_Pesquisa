using System;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Math;
using Accord.Math.Random;

class Program
{
    static void Main(string[] args)
    {
        // Definir os vetores de treinamento
        double[] X = { -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2 };
        double[] T = new double[X.Length];

        for (int i = 0; i < X.Length; i++)
        {
            T[i] = Math.Sin(X[i]);
        }

        // Transformar vetores de treinamento em matrizes de entrada e saída
        double[][] inputs = X.Reshape(X.Length, 1);
        double[][] outputs = T.Reshape(T.Length, 1);

        // Criar a rede neural com 1 neurônio na camada de entrada, 
        // duas camadas ocultas (3 e 2 neurônios) e 1 neurônio na camada de saída
        var network = new ActivationNetwork(new BipolarSigmoidFunction(), 1, 3, 2, 1);

        // Inicializar pesos aleatoriamente
        new NguyenWidrow(network).Randomize();

        // Criar o algoritmo de aprendizado
        var teacher = new BackPropagationLearning(network)
        {
            LearningRate = 0.1
        };

        // Treinar a rede neural
        double error;
        int epoch = 0;
        do
        {
            error = teacher.RunEpoch(inputs, outputs);
            epoch++;
            Console.WriteLine($"Epoch: {epoch}, Error: {error}");
        } while (error > 0.01 && epoch < 10000);

        // Testar a rede neural
        foreach (var input in inputs)
        {
            double[] output = network.Compute(input);
            Console.WriteLine($"Input: {input[0]}, Predicted: {output[0]}, Actual: {Math.Sin(input[0])}");
        }
    }
}

