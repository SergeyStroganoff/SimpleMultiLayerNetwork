package org.example;

import org.example.models.SimpleMultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Hello world!
 */
public class App {

    public static final int CLASSES = 3;
    public static final int FEATURES_INPUT = 4;

    public static void main(String[] args) {
        System.out.println("Current working directory: " + System.getProperty("user.dir"));
        String filePath = "iris.txt";
        String modelData = "model.zip";
        String dadaSetPath = "";
        SimpleMultiLayerNetwork baeldung = new SimpleMultiLayerNetwork();
        DataSet dataSet = baeldung.readerCSV(150,filePath, FEATURES_INPUT, CLASSES);
        baeldung.doNormalization(dataSet);
        DataSet[] dataSetParts = baeldung.splitToTrainRestPArts(dataSet, 0.65);
        var conf = baeldung.configure(FEATURES_INPUT, CLASSES);
        baeldung.createAndTrain(conf, dataSetParts, CLASSES);
        //  baeldung.createAndTrainEarlyStopping(conf,dataSet[0],);
    }
}
