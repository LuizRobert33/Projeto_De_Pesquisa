using System;
using Accord.Controls;
using Accord.Neuro;
using Accord.Neuro.Learning;
using System.Linq;

class Aproximacao_Seno_3
{
    // Método para desnormalizar valores
    public double Denormalize(double value, double min, double max)
    {
        return value * (max - min) + min;
    }

    static void Main(string[] args)
    {
        Aproximacao_Seno_3 app = new Aproximacao_Seno_3(); // Instanciar a classe para usar o método

        // Definir os vetores de treinamento
        double[] X = { -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2 };
        double[] T = new double[X.Length];

        for (int i = 0; i < X.Length; i++)
        {
            T[i] = Math.Sin(X[i]);
        }

        // Encontrar o mínimo e máximo dos vetores
        double minX = X.Min();
        double maxX = X.Max();
        double minT = T.Min();
        double maxT = T.Max();

        // Normalizar os vetores de entrada e saída
        double[] normalizedX = X.Select(v => (v - minX) / (maxX - minX)).ToArray();
        double[] normalizedT = T.Select(v => (v - minT) / (maxT - minT)).ToArray();

        // Transformar vetores de treinamento em matrizes de entrada e saída
        double[][] inputs = normalizedX.Select(v => new double[] { v }).ToArray();
        double[][] outputs = normalizedT.Select(v => new double[] { v }).ToArray();

        // Definir o número de folds (k) para a validação cruzada
        int k = 5;
        int foldSize = inputs.Length / k;
        double totalValidationError = 0;

        for (int fold = 0; fold < k; fold++)
        {
            // Separar dados de treino e validação para o fold atual
            double[][] trainingInputs = inputs.Where((_, idx) => idx < fold * foldSize || idx >= (fold + 1) * foldSize).ToArray();
            double[][] trainingOutputs = outputs.Where((_, idx) => idx < fold * foldSize || idx >= (fold + 1) * foldSize).ToArray();
            double[][] validationInputs = inputs.Skip(fold * foldSize).Take(foldSize).ToArray();
            double[][] validationOutputs = outputs.Skip(fold * foldSize).Take(foldSize).ToArray();

            // Criar a rede neural com 1 neurônio na camada de entrada, 
            // duas camadas ocultas com 3 e 2 neurônios e 1 neurônio na camada de saída
            var network = new ActivationNetwork(new SigmoidFunction(), 1, 3, 2, 1);

            // Inicializar pesos aleatoriamente
            new NguyenWidrow(network).Randomize();

            // Criar o algoritmo de aprendizado
            var teacher = new BackPropagationLearning(network)
            {
                LearningRate = 0.001 // Taxa de aprendizagem mais próxima testada 
            };

            // Treinar a rede neural
            double epochError;
            int epoca = 0;
            do
            {
                epochError = teacher.RunEpoch(trainingInputs, trainingOutputs);
                epoca++;
            } while (epochError > 0.01 && epoca < 10000);

            // Validar a rede neural e calcular o erro quadrático médio no fold atual
            double foldValidationError = 0.0;
            for (int i = 0; i < validationInputs.Length; i++)
            {
                double[] output = network.Compute(validationInputs[i]);

                // Desnormalizar os valores
                double predictedValue = app.Denormalize(output[0], minT, maxT);
                double expectedValue = app.Denormalize(validationOutputs[i][0], minT, maxT);

                // Calcular o erro quadrático
                double error = predictedValue - expectedValue;
                foldValidationError += error * error;
            }
            foldValidationError /= validationInputs.Length;
            totalValidationError += foldValidationError;

            Console.WriteLine($"Fold {fold + 1}/{k}, Erro Quadrático Médio de Validação: {foldValidationError}");
        }

        double meanValidationError = totalValidationError / k;
        Console.WriteLine($"Erro Quadrático Médio total (validação cruzada): {meanValidationError}");
    }
}
