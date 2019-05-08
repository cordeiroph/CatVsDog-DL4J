package catvsdog;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
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
        StatsStorage statsStorage = new FileStatsStorage(new File("/home/phrc/Documents/ml3 assigment/dl4j-log", "B-32-32-32.dl4j"));

        Random randNumGen = new Random(33);

//        FileNameLabelGenerator labelMaker = new FileNameLabelGenerator();
//        File trainData = new File("/home/phrc/Documents/ml3 assigment/images/train/");
        File trainData = new File("/home/phrc/Documents/ml3 assigment/Project3_files/data/train/");
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(trainSplit);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        DataNormalization imageScaler = new ImagePreProcessingScaler();
        imageScaler.fit(trainIter);
        trainIter.setPreProcessor(imageScaler);

        ComputationGraphConfiguration conf = getConf1(trainIter, channels, outputNum, height, width);

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new StatsListener(statsStorage, 1));
        uiServer.attach(statsStorage);

        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);
            trainIter.reset();
        }

        System.out.println("DONE");

    }

    public static ComputationGraphConfiguration getConf1(DataSetIterator trainIter, int channels, int outputNum, int height, int width){
        Map<String, InputPreProcessor> inputPreProcessorMap = new HashMap<>();
        inputPreProcessorMap.put("dense-64", new CnnToFeedForwardPreProcessor(23, 23, 3));

        ComputationGraphConfiguration.GraphBuilder conf = new NeuralNetConfiguration.Builder()
                .seed(33)
                .l2(0.0001)
                .updater(new Adam(0.005))
                .graphBuilder()
                .setInputTypes(InputType.convolutional(height, width, channels))
                .addInputs("input-net")


//                LAYER conv 1
                .addLayer("conv-32A", new ConvolutionLayer.Builder(2, 2)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build(), "input-net"
                )

//                LAYER conv 2

                .addLayer("conv-32B", new ConvolutionLayer.Builder(2, 2)
                        .stride(1, 1)
                        .nIn(32)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build(), "conv-32A")

                .addLayer("maxPooling-B", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .build(), "conv-32B")

//                LAYER conv 2

                .addLayer("conv-32C", new ConvolutionLayer.Builder(2, 2)
                        .stride(1, 1)
                        .nOut(32)
                        .nIn(32)
                        .activation(Activation.RELU)
                        .build(), "maxPooling-B")
                .addLayer("maxPooling-C", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .build(), "conv-32C")

//                LAYER dense 1

//                .addLayer("dense-64", new DenseLayer.Builder()
//                        .activation(Activation.RELU)
//                        .nOut(64)
//                        .nIn(23*23*3)
//                        .build(), "maxPooling-C")

//                LAYER dense out

                .addLayer("outputLayer",
                        new OutputLayer.Builder()
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .nOut(2)
                                .nIn(16928)
                                .build(), "maxPooling-C")
                .setOutputs("outputLayer")


//                .addLayer(7, new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
//                        .nOut(outputNum)
//                        .activation(Activation.SIGMOID)
//                        .build())
                ;
//
//        conf.setInputPreProcessors(inputPreProcessorMap);

        return conf.build();

    }

}