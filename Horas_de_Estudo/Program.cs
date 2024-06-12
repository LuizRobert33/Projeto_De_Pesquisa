using Accord.Controls;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics;
using Accord.Statistics.Kernels;
using System;

namespace ExamPrediction
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            // Dados de entrada: [Horas de Estudo, Frequência às Aulas]
            double[][] inputs =
            {
                new double[] { 2, 0.8 }, // Estudou 2 horas, frequência de 80%
                new double[] { 4, 0.6 }, // Estudou 4 horas, frequência de 60%
                new double[] { 1, 0.9 }, // Estudou 1 hora, frequência de 90%
                new double[] { 5, 0.4 }, // Estudou 5 horas, frequência de 40%
                new double[] { 3, 0.7 }, // Estudou 3 horas, frequência de 70%
                new double[] { 6, 0.2 }  // Estudou 6 horas, frequência de 20%
            };

            // Saída esperada: 1 - Aprovado, 0 - Reprovado
            int[] outputs =
            {
                1, // Aprovado
                1, // Aprovado
                1, // Aprovado
                0, // Reprovado
                1, // Aprovado
                0  // Reprovado
            };

            // Criar o algoritmo de aprendizado com o kernel Gaussiano
            var smo = new SequentialMinimalOptimization<Gaussian>()
            {
                Complexity = 100 // Criar uma SVM de margem dura
            };

            // Treinar a SVM com os dados fornecidos
            var svm = smo.Learn(inputs, outputs);

            // Fazer previsões usando a SVM treinada
            bool[] predictions = svm.Decide(inputs);

            // Calcular o erro de classificação
            double error = new AccuracyLoss(outputs).Loss(predictions);

            Console.WriteLine("Erro: " + error);

            // Mostrar os dados de treinamento e os resultados em gráficos
            ScatterplotBox.Show("Dados de Treinamento", inputs, outputs);
            ScatterplotBox.Show("Resultados da SVM", inputs, predictions.ToZeroOne());

            Console.ReadKey();
        }
    }
}
