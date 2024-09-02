using System;
using Accord.Neuro;
using Accord.Neuro.Learning;

class Aproximacao_Seno_3
{
    static void Main(string[] args)
    {
        // Definir os vetores de treinamento
        double[] X = { -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2 };
        double[] T = new double[X.Length];

        for (int i = 0; i < X.Length; i++)
        {
            T[i] = Math.Sin(X[i]);
        }

        // Encontrar o mínimo e máximo dos vetores
        double minX = double.MaxValue, maxX = double.MinValue;
        double minT = double.MaxValue, maxT = double.MinValue;

        for (int i = 0; i < X.Length; i++)
        {
            if (X[i] < minX) minX = X[i];
            if (X[i] > maxX) maxX = X[i];
            if (T[i] < minT) minT = T[i];
            if (T[i] > maxT) maxT = T[i];
        }

        // Normalizar os vetores de entrada e saída
        double[] normalizedX = new double[X.Length];
        double[] normalizedT = new double[T.Length];

        for (int i = 0; i < X.Length; i++)
        {
            normalizedX[i] = (X[i] - minX) / (maxX - minX);
            normalizedT[i] = (T[i] - minT) / (maxT - minT);
        }

        // Transformar vetores de treinamento em matrizes de entrada e saída
        double[][] inputs = new double[normalizedX.Length][];
        double[][] outputs = new double[normalizedT.Length][];
        for (int i = 0; i < normalizedX.Length; i++)
        {
            inputs[i] = new double[] { normalizedX[i] };
            outputs[i] = new double[] { normalizedT[i] };
        }

        // Criar a rede neural com 1 neurônio na camada de entrada, 
        // duas camadas ocultas com 3 e 2 neurônios e 1 neurônio na camada de saída
        var network = new ActivationNetwork(new SigmoidFunction(), 1, 3, 2, 1);

        // Inicializar pesos aleatoriamente
        new NguyenWidrow(network).Randomize();

        // Criar o algoritmo de aprendizado
        var teacher = new BackPropagationLearning(network)
        {
            LearningRate = 0.001 // Taxa de aprendizagem
        };

        // Definir o número de folds para a validação cruzada
        int k = 5;
        int foldSize = X.Length / k;

        double totalTrainingError = 0;
        double totalValidationError = 0;

        // Executar a validação cruzada
        for (int fold = 0; fold < k; fold++)
        {
            Console.WriteLine($"\nIniciando Fold {fold + 1}");

            // Preparar os dados de treinamento e validação para o fold atual
            double[][] foldTrainInputs = new double[X.Length - foldSize][];
            double[][] foldTrainOutputs = new double[X.Length - foldSize][];
            double[][] foldValidationInputs = new double[foldSize][];
            double[][] foldValidationOutputs = new double[foldSize][];

            int trainIndex = 0;
            int validationIndex = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                if (i >= fold * foldSize && i < (fold + 1) * foldSize)
                {
                    foldValidationInputs[validationIndex] = inputs[i];
                    foldValidationOutputs[validationIndex] = outputs[i];
                    validationIndex++;
                }
                else
                {
                    foldTrainInputs[trainIndex] = inputs[i];
                    foldTrainOutputs[trainIndex] = outputs[i];
                    trainIndex++;
                }
            }

            // Treinar a rede neural no fold atual
            double epochError;
            int epoca = 0;
            do
            {
                epochError = teacher.RunEpoch(foldTrainInputs, foldTrainOutputs);
                epoca++;
            } while (epochError > 0.01 && epoca < 10000);

            // Calcular o erro de treinamento
            double foldTrainingError = 0;
            for (int i = 0; i < foldTrainInputs.Length; i++)
            {
                double[] output = network.Compute(foldTrainInputs[i]);
                double predictedValue = Denormalize(output[0], minT, maxT);
                double actualValue = Math.Sin(Denormalize(foldTrainInputs[i][0], minX, maxX));
                foldTrainingError += Math.Pow(predictedValue - actualValue, 2);
            }
            foldTrainingError /= foldTrainInputs.Length;

            // Calcular o erro de validação
            double foldValidationError = 0;
            for (int i = 0; i < foldValidationInputs.Length; i++)
            {
                double[] output = network.Compute(foldValidationInputs[i]);
                double predictedValue = Denormalize(output[0], minT, maxT);
                double actualValue = Math.Sin(Denormalize(foldValidationInputs[i][0], minX, maxX));
                foldValidationError += Math.Pow(predictedValue - actualValue, 2);
            }
            foldValidationError /= foldValidationInputs.Length;

            // Acumular os erros
            totalTrainingError += foldTrainingError;
            totalValidationError += foldValidationError;

            Console.WriteLine($"Fold {fold + 1}: Erro de Treinamento: {foldTrainingError}, Erro de Validação: {foldValidationError}");
        }

        // Calcular a média dos erros de treinamento e validação
        double meanTrainingError = totalTrainingError / k;
        double meanValidationError = totalValidationError / k;

        Console.WriteLine($"\nErro Médio de Treinamento: {meanTrainingError}, Erro Médio de Validação: {meanValidationError}");

        // Manter o console aberto
        Console.ReadLine();
    }

    static double Denormalize(double value, double min, double max)
    {
        return value * (max - min) + min;
    }
}
