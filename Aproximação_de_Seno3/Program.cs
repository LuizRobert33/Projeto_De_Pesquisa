using System;
using System.IO;
using Accord.Controls;
using Accord.Neuro;
using Accord.Neuro.Learning;

class Aproximacao_Seno_3
{
    static void Main(string[] args)
    {
        // Definir os vetores de treinamento
        double[] X = { -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2 };
        double[] T = new double[X.Length];
        // Dados de Validação
        double[] validationInputs = { -1.3, -0.3, 0.5, 0.9, 1.3, 1.9 };
        double[] expectedOutputs = { Math.Sin(-1.3), Math.Sin(-0.3), Math.Sin(0.5), Math.Sin(0.9), Math.Sin(1.3), Math.Sin(1.9) };

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
            LearningRate = 0.001 // Taxa de aprendizagem mais próxima testada 
        };

        // Treinar a rede neural
        double trainingError;
        double epochError;
        int epoca = 0;
        do
        {
            epochError = teacher.RunEpoch(inputs, outputs);

            // Cálculo manual do erro entre o y real e o y previsto
            trainingError = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] output = network.Compute(inputs[i]);
                double predictedValue = Denormalize(output[0], minT, maxT);
                double actualValue = Math.Sin(Denormalize(inputs[i][0], minX, maxX));
                trainingError += Math.Pow(predictedValue - actualValue, 2);
            }
            trainingError /= inputs.Length;

            epoca++;
            // Erro Quadrático Médio (MSE): Mede a média dos quadrados das diferenças entre os valores reais e os valores previstos.
            // Erro Médio de Treinamento: Média dos erros absolutos entre os valores reais e previstos durante o treinamento.
            Console.WriteLine($"Epoca: {epoca}, Erro Médio no Treinamento: {epochError}, Erro Quadrático Médio: {trainingError}");

        } while (epochError > 0.01 && epoca < 10000);

        // Criar um arquivo txt com os resultados do treinamento
        using (StreamWriter writer = new StreamWriter("resultados_aproximacao_de_Seno.txt"))
        {
            writer.WriteLine("Input\tPredicted\tActual");
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] output = network.Compute(inputs[i]);

                // Desnormalizar os valores
                double actualValue = Math.Sin(Denormalize(inputs[i][0], minX, maxX));
                double predictedValue = Denormalize(output[0], minT, maxT);

                writer.WriteLine($"{Denormalize(inputs[i][0], minX, maxX)}\t{predictedValue}\t{actualValue}");
                Console.WriteLine($"Input: {Denormalize(inputs[i][0], minX, maxX)}, Predicted: {predictedValue}, Actual: {actualValue}");
            }
        }

        Console.WriteLine("Resultado salvo em: 'resultados_aproximacao_de_Seno.txt'");

        // Normalizar dados de validação
        double[] normalizedValidationInputs = new double[validationInputs.Length]; 
        for (int i = 0; i < validationInputs.Length; i++)
        {
            normalizedValidationInputs[i] = (validationInputs[i] - minX) / (maxX - minX);
        }

        // Validar a rede neural
        using (StreamWriter writer = new StreamWriter("resultados_validacao.txt"))
        {
            writer.WriteLine("Input\tPredicted\tExpected");
            double squaredErrorSum = 0.0;
            for (int i = 0; i < validationInputs.Length; i++)
            {
                double[] output = network.Compute(new double[] { normalizedValidationInputs[i] });

                // Desnormalizar os valores
                double predictedValue = Denormalize(output[0], minT, maxT);
                double expectedValue = expectedOutputs[i];

                // Calcular o erro quadrático para validação 
                double error = predictedValue - expectedValue;
                squaredErrorSum += error * error;

                writer.WriteLine($"{validationInputs[i]}\t{predictedValue}\t{expectedValue}");
                Console.WriteLine($"Input: {validationInputs[i]}, Predicted: {predictedValue}, Expected: {expectedValue}");
            }

            double meanSquaredErrorValidation = squaredErrorSum / validationInputs.Length;
            // Console.WriteLine($"Erro Quadrático Médio de Validação: {meanSquaredErrorValidation}");
        }

        // Console.WriteLine("Resultado de validação salvo em: 'resultados_validacao.txt'");

        // Preparar os dados para o gráfico de treinamento
        double[] predicted = new double[X.Length];
        for (int i = 0; i < X.Length; i++)
        {
            double[] output = network.Compute(new double[] { (X[i] - minX) / (maxX - minX) });
            predicted[i] = Denormalize(output[0], minT, maxT);
        }

        // Criar o gráfico de treinamento
        ScatterplotBox.Show("Gráfico de Treinamento - Valores Reais", X, T);
        ScatterplotBox.Show("Gráfico de Treinamento - Valores Da Rede", X, predicted);

        // Preparar os dados para o gráfico de validação
        double[] predictedValidation = new double[validationInputs.Length];
        for (int i = 0; i < validationInputs.Length; i++)
        {
            double[] output = network.Compute(new double[] { normalizedValidationInputs[i] });
            predictedValidation[i] = Denormalize(output[0], minT, maxT);
        }
    }

    static double Denormalize(double value, double min, double max)
    {
        return value * (max - min) + min;
    }
}