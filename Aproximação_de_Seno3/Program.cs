using System;
using System.Drawing;
using System.Windows.Forms;
using System.IO;
using Accord.Controls;
using Accord.Neuro;
using Accord.Neuro.Learning;
using Accord.Math.Random;

class Aproximacao_Seno_3
{
    static void Main(string[] args)
    {
        // Definir os vetores 
        double[] X = { -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2 };
        double[] T = new double[X.Length];

        for (int i = 0; i < X.Length; i++)
        {
            T[i] = Math.Sin(X[i]);
        }

        // Transformar vetores de treinamento em matrizes de entrada e saída
        //Transforma os vetores de entrada e saída em matrizes bidimensionais, onde cada elemento é um vetor unidimensional. 
        double[][] inputs = new double[X.Length][];
        double[][] outputs = new double[T.Length][];
        for (int i = 0; i < X.Length; i++)
        {
            inputs[i] = new double[] { X[i] };
            outputs[i] = new double[] { T[i] };
        }

        // Criar a rede neural com 1 neurônio na camada de entrada, 
        // duas camadas ocultas com 3 e 2 neurônios e 1 neurônio na camada de saída
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
        int epoca = 0;
        do
        {
            error = teacher.RunEpoch(inputs, outputs);
            epoca++;
            Console.WriteLine($"Epoca: {epoca}, Erro: {error}");
        } while (error > 0.01 && epoca < 10000);

        // Criar um arquivo txt
        using (StreamWriter writer = new StreamWriter("resultados_aproximacao_de_Seno.txt"))
        {
            writer.WriteLine("Input\tPredicted\tActual");  // Resultado , Previsto , Real
            // Testar a rede neural
            foreach (var input in inputs)
            {
                double[] output = network.Compute(input);
                writer.WriteLine($"{input[0]}\t{output[0]}\t{Math.Sin(input[0])}");
                Console.WriteLine($"Input: {input[0]}, Predicted: {output[0]}, Actual: {Math.Sin(input[0])}"); // Resultado , Previsto , Real
            }
        }

        Console.WriteLine("Resultado salvo em: 'resultados_aproximacao_de_Seno.txt'");

        // Preparar os dados para o gráfico
        double[] predicted = new double[X.Length];
        for (int i = 0; i < X.Length; i++)
        {
            double[] output = network.Compute(new double[] { X[i] });
            predicted[i] = output[0];
        }

        // Criar o gráfico
        ShowGraph(X, T, predicted);
    }

    static void ShowGraph(double[] x, double[] actual, double[] predicted)
    {
        // Mostrar os dados reais
        ScatterplotBox.Show("Valores Reais", x, actual);

        // Mostrar os dados previstos
        ScatterplotBox.Show("Valores Da Rede", x, predicted);
    }
}

