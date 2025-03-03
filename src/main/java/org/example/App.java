package org.example;

import org.example.models.SimpleMultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Hello in LLM world!
 * Precision, Recall, F1-Score
 * Precision (98.67%) — насколько предсказания правильны среди всех предсказанных примеров. Высокий показатель = мало ложных срабатываний.
 * Recall (97.62%) — сколько реальных примеров модель правильно определила. Высокий показатель = мало пропущенных примеров.
 * F1-Score (98.09%) — баланс precision и recall, хороший показатель.
 */
public class App {

    public static final int CLASSES = 3;
    public static final int FEATURES_INPUT = 4;

    public static void main(String[] args) {
        System.out.println("Current working directory: " + System.getProperty("user.dir"));
        String filePath = "iris.txt";
        String modelDataSavePath = "model.zip";
        String dadaSetPath = "";
        SimpleMultiLayerNetwork baeldung = new SimpleMultiLayerNetwork();
        DataSet dataSet = baeldung.readerCSV(150, filePath, FEATURES_INPUT, CLASSES);
        baeldung.doNormalization(dataSet);
        DataSet[] dataSetParts = baeldung.splitToTrainRestPArts(dataSet, 0.65);
        var conf = baeldung.configure(FEATURES_INPUT, CLASSES);
        var model = baeldung.createAndTrain(conf, dataSetParts, CLASSES, 500);
        baeldung.saveModel(model, modelDataSavePath);
        //  baeldung.createAndTrainEarlyStopping(conf,dataSet[0],);
    }
}
