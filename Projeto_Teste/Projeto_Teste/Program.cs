using System;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Math;

class Program
{
    static void Main()
    {
        // Dados de entrada e saída
        double[][] inputs = {
            new double[] { 840.51, 150.91, 91.94 },
            new double[] { 1054.40, 150.81, 91.88 },
            new double[] { 840.51, 150.64, 91.69 },
            new double[] { 599.23, 150.56, 91.84 },
            new double[] { 511.37, 153.02, 93.95 },
            new double[] { 789.87, 152.79, 94.33 }
            // Adiconar mais dados caso seja necessario
        };

        double[][] outputs = {
            new double[] { 91.88 },
            new double[] { 91.69 },
            new double[] { 91.84 },
            new double[] { 93.95 },
            new double[] { 94.33 },
            new double[] { 94.08 }
            //  Adiconar mais dados caso seja necessario
        };

        // Criação da rede neural com 3 entradas, uma camada oculta de 5 neurônios e 1 neurônio de saída
        var network = new ActivationNetwork(
            function: new SigmoidFunction(),
            inputsCount: 3,  // 3 entradas: vazão e nível do reservatório
            neuronsCount: 5  // 5 neurônios na camada oculta
        );

        // Inicialização dos pesos da rede com o algoritmo NguyenWidrow
        new NguyenWidrow(network).Randomize();

        // Configuração do algoritmo de aprendizado
        var teacher = new LevenbergMarquardtLearning(network)
        {
            LearningRate = 0.1
        };

        // Número de épocas de treinamento
        int epochs = 1000;

        // Treinamento da rede neural
        for (int i = 0; i < epochs; i++)
        {
            double error = teacher.RunEpoch(inputs, outputs); // Treina uma época e retorna o erro
            Console.WriteLine("Época: " + i + ", Erro: " + error);

            // Condição de parada: se o erro estiver abaixo do limite desejado, interrompe o treinamento
            if (error < 0.001)
                break;
        }

        // Dados de entrada para teste
        double[] testInput = { 789.87, 150.56, 91.84 };

        // Fazendo uma previsão com a rede treinada
        double[] result = network.Compute(testInput);

        // Exibindo o resultado
        Console.WriteLine("Previsão do nível do reservatório Santa Rosa: " + result[0]);
    }
}
