package org.example.models;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * <a href="https://www.baeldung.com/deeplearning4j">...</a>
 */

public class SimpleMultiLayerNetwork {


    public DataSet readerCSV(int batchSize, String path, int featuresInput, int classes) {
        DataSet allData = null;
        //читаем файл с данными
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(new ClassPathResource(path).getFile()));
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, featuresInput, classes);
            allData = iterator.next();
            allData.shuffle();

            // …
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        return allData;
    }

    /**
     * Another thing we should do with the data before training is to normalize it. The normalization is a two-phase process:
     * <p>
     * gathering of some statistics about the data (fit)
     * changing (transform) the data in some way to make it uniform
     *
     * @param allData
     */
    public void doNormalization(DataSet allData) {
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);
    }

    public DataSet[] splitToTrainRestPArts(DataSet allData, double trainPartSize) {
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(trainPartSize);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        return new DataSet[]{trainingData, testData};
    }

    /**
     * Next, we create a network of dense (also known as fully connect) layers.
     * The first layer should contain the same amount of nodes as the columns in the training data (4).
     * The second dense layer will contain three nodes. This is the value we can variate,
     * but the number of outputs in the previous layer has to be the same.
     * The final output layer should contain the number of nodes matching the number of classes (3).
     * The structure of the network is shown in the picture:
     */

    public MultiLayerConfiguration configure(int inputFeatures, int classes) {
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                // Устанавливаем функцию активации по умолчанию для всех слоев
                .activation(Activation.RELU)
                // Возможные альтернативы:
                // - Activation.SIGMOID (подходит для бинарной классификации, но есть проблема с исчезающим градиентом)
                // - Activation.TANH (улучшенная версия сигмоиды, подходит для неглубоких сетей)
                // - Activation.LEAKYRELU (альтернатива ReLU для избежания "затыка" нейронов)

                // Инициализация весов: XAVIER_UNIFORM подходит для стабильного старта обучения
                .weightInit(WeightInit.XAVIER_UNIFORM)
                // Возможные альтернативы:
                // - WeightInit.XAVIER (вариант без равномерного распределения, но тоже стабилен)
                // - WeightInit.RELU (рекомендуется, если используем ReLU)
                // - WeightInit.ZERO (обнуление весов — не рекомендуется, так как слои будут одинаковыми)

                // Выбираем метод обновления весов: SGD (стохастический градиентный спуск)
                .updater(new Sgd(0.01))
                // Возможные альтернативы:
                // - new Adam(0.001) (адаптивный алгоритм, часто лучше работает на сложных данных)
                // - new Nesterovs(0.01, 0.9) (вариант SGD с моментом для ускорения обучения)
                // - new RMSProp(0.001) (подходит для рекуррентных нейронных сетей)

                .list() // Начинаем определять список слоев сети

                // Первый скрытый слой - полносвязный (DenseLayer)
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputFeatures).nOut(10) // FEATURES_COUNT - количество входных признаков, 10 нейронов в слое
                        .l2(0.001) // Добавляем L2-регуляризацию, чтобы уменьшить переобучение
                        .build())
                // Возможные альтернативы:
                // - .nOut(5) (уменьшим число нейронов — менее выразительная модель)
                // - .nOut(20) (увеличим число нейронов — риск переобучения, но больше возможностей)
                // - Можно добавить `dropOut(0.5)`, чтобы отключать 50% нейронов во время обучения для регуляризации

                // Второй скрытый слой - тоже полно связный
                .layer(1, new DenseLayer.Builder()
                        .nIn(10).nOut(6) // Получает 10 входов, отдает 6 выходов
                        .l2(0.001) // Регуляризация для контроля переобучения
                        .build())
                // Альтернативы:
                // - Можно убрать второй слой, если данных мало и модель слишком сложна
                // - Можно использовать ConvolutionLayer, если данные — изображения

                // Выходной слой - отвечает за предсказание класса
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX) // Softmax превращает выходы в вероятности классов
                        .nIn(6).nOut(classes) // Входные 6 нейронов, выходных - столько, сколько классов (3 для Iris)
                        .l2(0.001) // Регуляризация для выхода
                        .build())
                // Альтернативы:
                // - LossFunctions.MSE (используется для регрессии, но не подходит для классификации)
                // - LossFunctions.XENT (кросс-энтропия, аналогична NEGATIVELOGLIKELIHOOD, но проще)

                .build(); // Завершаем конфигурацию сети
        return configuration;
    }

    public MultiLayerNetwork createAndTrain(MultiLayerConfiguration configuration, DataSet[] dataSets, int classes, int epochs) {
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        DataSet trainData = dataSets[0];
        DataSet testData = dataSets[1];

        // Количество эпох (настраиваемый параметр)
        Evaluation eval = null;
        for (int i = 0; i < epochs; i++) {
            trainData.shuffle(); // Перемешивание данных перед каждой эпохой
            model.fit(trainData);

            // Оценка точности после каждой эпохи
            INDArray output = model.output(testData.getFeatures());
            eval = new Evaluation(classes);
            eval.eval(testData.getLabels(), output);

            System.out.println("Эпоха " + (i + 1) + ":");
            System.out.println(eval.accuracy());
            if (Double.compare(eval.accuracy(), 0.98) > 0) {
                break;
            }

        }
        if (eval != null) {
            System.out.println(eval.stats());
        } else {
            System.out.println("Training process fault");
        }
        return model;
    }

    public void saveModel(MultiLayerNetwork model, String filePath) {
        try {
            File modelFile = new File(filePath);
            ModelSerializer.writeModel(model, modelFile, true);
            System.out.println("Модель сохранена в: " + filePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public MultiLayerNetwork loadModel(String filePath) {
        try {
            File modelFile = new File(filePath);
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            System.out.println("Модель загружена из: " + filePath);
            return model;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public void createAndTrainEarlyStopping(MultiLayerConfiguration configuration, DataSetIterator trainData, DataSetIterator testData) {
        // Директория для сохранения модели
        File directory = new File("saved_model/");
        // Конфигурация ранней остановки
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(30)) // Максимум 30 эпох
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) // Остановить, если обучение идет > 20 минут
                .scoreCalculator(new DataSetLossCalculator(testData, true)) // Оценка качества на тестовом датасете
                .evaluateEveryNEpochs(1) // Оценивать каждую эпоху
                .modelSaver(new LocalFileModelSaver(directory)) // Сохранение лучшей модели
                .build();
        // Тренер с ранней остановкой
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, configuration, trainData);
        // Запуск обучения
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        // Вывод результатов обучения
        System.out.println("Причина остановки: " + result.getTerminationReason());
        System.out.println("Детали остановки: " + result.getTerminationDetails());
        System.out.println("Всего эпох: " + result.getTotalEpochs());
        System.out.println("Лучшая эпоха: " + result.getBestModelEpoch());
        System.out.println("Лучшая оценка модели: " + result.getBestModelScore());
        // Получаем лучшую модель
        MultiLayerNetwork bestModel = result.getBestModel();
    }
}
