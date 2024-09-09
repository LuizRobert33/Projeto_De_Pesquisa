using System;
using Accord.Controls;
using Accord.Neuro;
using Accord.Neuro.Learning;

class Aproximacao_Seno_3
{
    // Método para desnormalizar valores, convertendo valores normalizados de volta ao intervalo original
    public double Denormalize(double value, double min, double max)
    {
        return value * (max - min) + min;
    }

    static void Main(string[] args)
    {
        Aproximacao_Seno_3 app = new Aproximacao_Seno_3(); // Instanciar a classe para usar o método de desnormalização

        // Definir os vetores de treinamento com entradas e saídas esperadas
        double[] X = { -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2 };
        double[] T = new double[X.Length];

        // Dados de Validação
        double[] validationInputs = { -1.3, -0.3, 0.5, 0.9, 1.3, 1.9 };
        double[] expectedOutputs = { Math.Sin(-1.3), Math.Sin(-0.3), Math.Sin(0.5), Math.Sin(0.9), Math.Sin(1.3), Math.Sin(1.9) };

        // Calcular as saídas esperadas T com base na função seno
        for (int i = 0; i < X.Length; i++)
        {
            T[i] = Math.Sin(X[i]);
        }

        // Encontrar o mínimo e máximo dos vetores de entrada e saída para normalização
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

        // Inicializar pesos aleatoriamente para começar o treinamento
        new NguyenWidrow(network).Randomize();

        // Criar o algoritmo de aprendizado para a rede neural
        var teacher = new BackPropagationLearning(network)
        {
            LearningRate = 0.001
        };

        // Treinar a rede neural
        double trainingError;
        int epoca = 0;
        do
        {
            trainingError = teacher.RunEpoch(inputs, outputs);

            // Cálculo do erro quadrático médio entre o valor real e o previsto durante o treinamento
            double squaredErrorSum = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] output = network.Compute(inputs[i]); // Computa a saída da rede para a entrada normalizada
                double predictedValue = app.Denormalize(output[0], minT, maxT); // Desnormaliza o valor previsto
                double actualValue = Math.Sin(app.Denormalize(inputs[i][0], minX, maxX)); // Calcula o valor real
                double error = predictedValue - actualValue; // Calcula o erro
                squaredErrorSum += error * error; // Acumula o erro quadrático
            }
            double trainingSquaredError = squaredErrorSum / inputs.Length; // Calcula o erro quadrático médio

            epoca++;
            Console.WriteLine($"Época: {epoca}, Erro Quadrático Médio: {trainingSquaredError}");

        } while (trainingError > 0.01 && epoca < 10000); // Continua o treinamento até atingir o erro mínimo ou o número máximo de épocas

        // Normalizar dados de validação
        double[] normalizedValidationInputs = new double[validationInputs.Length];
        for (int i = 0; i < validationInputs.Length; i++)
        {
            normalizedValidationInputs[i] = (validationInputs[i] - minX) / (maxX - minX);
        }

        // Validar a rede neural e calcular o erro quadrático médio
        double squaredErrorSumValidation = 0.0;
        for (int i = 0; i < validationInputs.Length; i++)
        {
            double[] output = network.Compute(new double[] { normalizedValidationInputs[i] }); // Computa a saída da rede para a entrada de validação

            // Desnormalizar os valores
            double predictedValue = app.Denormalize(output[0], minT, maxT); // Desnormaliza o valor previsto
            double expectedValue = expectedOutputs[i]; // Obtém o valor esperado

            // Calcular o erro quadrático para validação 
            double error = predictedValue - expectedValue; // Calcula o erro
            squaredErrorSumValidation += error * error; // Acumula o erro quadrático

            Console.WriteLine($"Input: {validationInputs[i]}, Predicted: {predictedValue}, Expected: {expectedValue}");
        }

        double meanSquaredErrorValidation = squaredErrorSumValidation / validationInputs.Length; // Calcula o erro quadrático médio para validação
        Console.WriteLine($"Erro Quadrático Médio de Validação: {meanSquaredErrorValidation}");

        // Preparar os dados para o gráfico de treinamento
        double[] predicted = new double[X.Length];
        for (int i = 0; i < X.Length; i++)
        {
            double[] output = network.Compute(new double[] { (X[i] - minX) / (maxX - minX) }); // Computa a saída da rede para a entrada normalizada
            predicted[i] = app.Denormalize(output[0], minT, maxT); // Desnormaliza o valor previsto
        }

        // Criar o gráfico de treinamento
        ScatterplotBox.Show("Gráfico de Treinamento - Valores Reais", X, T); // Gráfico dos valores reais
        ScatterplotBox.Show("Gráfico de Treinamento - Valores da Rede", X, predicted); // Gráfico dos valores previstos pela rede

        // Manter o terminal aberto para visualizar os resultados
        Console.WriteLine("Pressione Enter para fechar...");
        Console.ReadLine();
    }
}
