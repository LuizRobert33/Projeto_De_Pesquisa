using System;
using Accord.Neuro;
using Accord.Neuro.Learning;

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
        };

        double[][] outputs = {
            new double[] { 91.88 },
            new double[] { 91.69 },
            new double[] { 91.84 },
            new double[] { 93.95 },
            new double[] { 94.33 },
            new double[] { 94.08 }
        };

        // Normalização dos dados de entrada e saída
        NormalizeData(ref inputs);
        NormalizeData(ref outputs);

        // Verificação dos tamanhos das entradas e saídas
        if (inputs.Length != outputs.Length)
        {
            Console.WriteLine("Erro: O número de amostras de entrada não corresponde ao número de saídas.");
            return;
        }

        // Criação da rede neural com 3 entradas, uma camada oculta de 10 neurônios e 1 neurônio de saída
        var network = new ActivationNetwork(
            function: new SigmoidFunction(),
            inputsCount: 3,  // 3 entradas
            neuronsCount: new int[] { 10, 1 } // 10 neurônios na camada oculta, 1 na camada de saída
        );

        // Inicialização dos pesos da rede com o algoritmo Nguyen-Widrow
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

        // Normaliza os dados de entrada para teste
        NormalizeTestData(ref testInput);

        // Fazendo uma previsão com a rede treinada
        double[] result = network.Compute(testInput);

        // Exibindo o resultado
        Console.WriteLine("Previsão do nível do reservatório Santa Rosa: " + result[0]);
    }

    // Função de normalização de dados (normaliza para o intervalo 0-1)
    static void NormalizeData(ref double[][] data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            double max = FindMax(data[i]);
            double min = FindMin(data[i]);
            for (int j = 0; j < data[i].Length; j++)
            {
                data[i][j] = (data[i][j] - min) / (max - min);
            }
        }
    }

    // Função de normalização dos dados de entrada para teste
    static void NormalizeTestData(ref double[] data)
    {
        double max = FindMax(data);
        double min = FindMin(data);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (data[i] - min) / (max - min);
        }
    }

    // Função auxiliar para encontrar o valor máximo de um vetor
    static double FindMax(double[] data)
    {
        double max = data[0];
        foreach (double val in data)
        {
            if (val > max)
            {
                max = val;
            }
        }
        return max;
    }

    // Função auxiliar para encontrar o valor mínimo de um vetor
    static double FindMin(double[] data)
    {
        double min = data[0];
        foreach (double val in data)
        {
            if (val < min)
            {
                min = val;
            }
        }
        return min;
    }
}
