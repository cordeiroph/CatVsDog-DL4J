package catvsdog;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Main {


    public static void main(String[] args) throws Exception {
        int height = 96;
        int width = 96;
        int channels = 3;
        int outputNum = 2;
        int batchSize = 32;
        int nEpochs = 10;
//

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File("/home/phrc/Documents/ml3 assigment/dl4j-log", "32-32-32.dl4j"));

        Random randNumGen = new Random(33);

        FileNameLabelGenerator labelMaker = new FileNameLabelGenerator();
        File trainData = new File("/home/phrc/Documents/ml3 assigment/images/train/");
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(trainSplit);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        DataNormalization imageScaler = new ImagePreProcessingScaler();
        imageScaler.fit(trainIter);
        trainIter.setPreProcessor(imageScaler);

        Map<Integer, Double> learningRateSchedule = new HashMap<>();
        learningRateSchedule.put(0, 0.06);
        learningRateSchedule.put(200, 0.05);
        learningRateSchedule.put(600, 0.028);
        learningRateSchedule.put(800, 0.0060);
        learningRateSchedule.put(1000, 0.001);


        MultiLayerConfiguration conf = getConf1(learningRateSchedule, channels, outputNum, height, width);

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(
                new StatsListener(statsStorage, 1));
        uiServer.attach(statsStorage);

        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);
            trainIter.reset();
        }

        System.out.println("DONE");

    }

    public static MultiLayerConfiguration getConf1(Map<Integer, Double> learningRateSchedule, int channels, int outputNum, int height, int width){
        return new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(new MapSchedule(ScheduleType.EPOCH, learningRateSchedule)))
                .list()

//                LAYER conv 1
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build()
                )
//                LAYER conv 2

                .layer(new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .build())

//                LAYER conv 2

                .layer(new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .build())

//                LAYER dense 1


                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(64)
                        .build())

//                LAYER dense out

                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(1)
                        .build())

                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                        .nOut(outputNum)
                        .activation(Activation.SIGMOID)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
                .build();

    }

}