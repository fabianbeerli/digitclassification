package ch.zhaw.deeplearningjava.digit;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;


public final class Training {

    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 2;

    public static void main(String[] args) throws IOException, TranslateException {
        Path modelDir = Paths.get("models");

        // create ImageFolder dataset from directory
        // ImageFolder dataset = initDataset("ut-zap50k-images-square");
        ImageFolder dataset = initDataset("mnist_training");
        // Split the dataset set into training dataset and validate dataset
        RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);

        Loss loss = Loss.softmaxCrossEntropyLoss();
        TrainingConfig config = setupTrainingConfig(loss);

        Model model = Models.getModel();
        Trainer trainer = model.newTrainer(config);
        trainer.setMetrics(new Metrics());

        // Resize MNIST images to fit the input shape of the chosen model
        Shape inputShape = new Shape(1, 3, Models.IMAGE_HEIGHT, Models.IMAGE_WIDTH);

        trainer.initialize(inputShape);

        EasyTrain.fit(trainer, EPOCHS, datasets[0], datasets[1]);

        TrainingResult result = trainer.getTrainingResult();
        model.setProperty("Epoch", String.valueOf(EPOCHS));
        model.setProperty("Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
        model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

        model.save(modelDir, Models.MODEL_NAME);

        // save labels into model directory
        Models.saveSynset(modelDir, dataset.getSynset());
    }

    private static ImageFolder initDataset(String datasetRoot)
        throws IOException, TranslateException {
        ImageFolder dataset = ImageFolder.builder()
            // retrieve the data
            .setRepositoryPath(Paths.get(datasetRoot))
            .optMaxDepth(10)
            .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
            .addTransform(new ToTensor())
            // random sampling; don't process the data in order
            .setSampling(BATCH_SIZE, true)
            .build();

        dataset.prepare();
        return dataset;
        }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }


}
