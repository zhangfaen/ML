package faen;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class Main {
    private static int BLACK_WHITE_THRESHOLD = 100;

    private static void printImage(int[][] pic) {
        // System.out.println("total:" + total);
        for (int i = 0; i < pic.length; i++) {
            for (int j = 0; j < pic[0].length; j++) {
                if (pic[i][j] > 0) {
                    System.out.print(1);
                } else {
                    System.out.print(0);
                }
            }
            System.out.println();
        }
        System.out.println("---------");
    }

    private static int getFeatureValue(int raw) {
        if (raw >= BLACK_WHITE_THRESHOLD) {
            return 1;
        }
        return 0;
    }

    private static List<Instance> readTestCsv(Instances instances) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(
                "/Users/zhangfaen/dev/ml/kaggle/digit_recognizer/data/test.csv"));
        ArrayList<Instance> ret = new ArrayList<Instance>();
        int total = 0;
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            if (total++ == 0) {
                // The first line is header.
                continue;
            }
            // if (total > 100) {
            // break;
            // }
            String[] lineArray = line.split(",");
            int m = 28;
            int n = 28;
            int[][] image = new int[m][n];
            Instance instance = new Instance(m * n);
            instance.setDataset(instances);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    image[i][j] = getFeatureValue(Integer.parseInt(lineArray[i * m + j]));
                    instance.setValue(i * m + j, image[i][j]);
                }
            }
            if (total < 10) {
                printImage(image);
            }
            ret.add(instance);
        }
        br.close();
        return ret;
    }

    private static FastVector buildAttributes() {
        FastVector ret = new FastVector();
        for (int i = 0; i < 28 * 28; i++) {
            ret.addElement(new Attribute("pixel" + i));
        }
        FastVector classValue = new FastVector();
        for (int i = 0; i < 10; i++) {
            classValue.addElement("" + i);
        }
        ret.addElement(new Attribute("theClasses", classValue));
        return ret;
    }

    private static Instances readTrainCsv() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(
                "/Users/zhangfaen/dev/ml/kaggle/digit_recognizer/data/train.csv"));
        int total = 0;
        FastVector attrs = buildAttributes();
        Instances instances = new Instances("data", attrs, 100);
        instances.setClassIndex(attrs.size() - 1);
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }
            if (total++ == 0) {
                // The first line is header.
                continue;
            }
            // if (total >= 3000) {
            // break;
            // }
            String[] lineArray = line.split(",");
            int m = 28;
            int n = 28;
            Instance instance = new Instance(m * n + 1);
            instance.setDataset(instances);
            int[][] image = new int[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    image[i][j] = getFeatureValue(Integer.parseInt(lineArray[i * m + j + 1]));
                    instance.setValue(i * m + j, image[i][j]);
                }
            }
            instance.setClassValue(lineArray[0]);
            instances.add(instance);
            // if (total % 3001 == 1) {
            // printImage(image);
            // }
        }
        br.close();
        return instances;
    }

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
        Instances instances = readTrainCsv();
        RandomForest model = new RandomForest();
        model.setOptions(Utils.splitOptions("-I 30 -K 30"));
        model.buildClassifier(instances);
        System.out.println(model.getRevision());
        System.out.println(Arrays.toString(model.getOptions()));
        System.out.println(model.toString());

        BufferedWriter bw = new BufferedWriter(new FileWriter(
                "/Users/zhangfaen/dev/ml/kaggle/digit_recognizer/data/result.csv"));
        bw.write("ImageId,Label\n");
        int index = 0;
        for (Instance instance : readTestCsv(instances)) {
            double d = model.classifyInstance(instance);
            int i = (int) d;
            if (i < 0 || i >= 10) {
                throw new Exception();
            }
            index++;
            bw.write(index + "," + i + "\n");
        }
        bw.close();
        // model.classifyInstance()
    }
}
