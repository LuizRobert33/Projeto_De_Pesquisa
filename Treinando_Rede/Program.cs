using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Neuro.Networks;
using Accord.Math;
using System;

namespace NeuralNetworkExample
{
    class Program
    {
        static void Main(string[] args)
        {
            // Dados de entrada: [Horas de Estudo, Frequência às Aulas]
            double[][] inputs =
            {
                new double[] { 2, 0.8 },
                new double[] { 4, 0.6 },
                new double[] { 1, 0.9 },
                new double[] { 5, 0.4 },
                new double[] { 3, 0.7 },
                new double[] { 6, 0.2 }
            };

            // Saída esperada: 1 - Aprovado, 0 - Reprovado
            double[][] outputs =
            {
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 0 }
            };

            // Criar uma rede neural com 2 neurônios na camada de entrada,
            // 3 neurônios na camada oculta e 1 neurônio na camada de saída
            var network = new ActivationNetwork(
                new SigmoidFunction(), // Função de ativação
                2, // Número de neurônios na camada de entrada
                3, // Número de neurônios na camada oculta
                1  // Número de neurônios na camada de saída
            );

            // Inicializar os pesos da rede
            new NguyenWidrow(network).Randomize();

            // Criar o algoritmo de aprendizado
            var teacher = new BackPropagationLearning(network)
            {
                LearningRate = 0.1,
                Momentum = 0.9
            };

            // Treinar a rede neural
            int epoch = 0;
            double error;
            do
            {
                error = teacher.RunEpoch(inputs, outputs);
                epoch++;
                Console.WriteLine($"Epoch: {epoch}, Error: {error}");
            }
            while (error > 0.01 && epoch < 1000);

            // Fazer previsões usando a rede neural treinada
            foreach (var input in inputs)
            {
                double[] output = network.Compute(input);
                Console.WriteLine($"Input: {string.Join(", ", input)}, Output: {output[0]:F4}");
            }
        }
    }
}
