package faen;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Utils;

public class Main {
    private static void crossEvaluation(Classifier model, Instances instances) throws Exception {
        Evaluation eval = new Evaluation(instances);
        eval.crossValidateModel(model, instances, 10, new Random(1));
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toSummaryString(true));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
        System.out.println(model.toString());
    }

    public static void main(String[] args) throws Exception {
        BufferedReader br1 = new BufferedReader(new FileReader(
                "/Users/zhangfaen/dev/ml/kaggle/titanic/data/train.arff"));
        Instances instances = new Instances(br1);
        instances.setClassIndex(8);
        br1.close();
        RandomForest model = new RandomForest();
        model.setOptions(Utils.splitOptions("-I 30 -K 6"));
        model.buildClassifier(instances);

        System.out.println(model.getRevision());
        System.out.println(Arrays.toString(model.getOptions()));
        System.out.println(model.toString());
        crossEvaluation(model, instances);

        BufferedReader br2 = new BufferedReader(new FileReader(
                "/Users/zhangfaen/dev/ml/kaggle/titanic/data/test.arff"));
        Instances instancesForTesting = new Instances(br2);
        instancesForTesting.setClassIndex(8);
        br2.close();

        BufferedWriter bw = new BufferedWriter(new FileWriter(
                "/Users/zhangfaen/dev/ml/kaggle/titanic/data/resulve.csv"));
        bw.write("PassengerId,Survived\n");
        for (int i = 0; i < instancesForTesting.numInstances(); i++) {
            // System.out.println(instancesForTesting.instance(i).toString());
            double f = model.classifyInstance(instancesForTesting.instance(i));
            bw.write("" + (892 + i) + "," + (int) f + "\n");
        }
        bw.close();
    }
}
