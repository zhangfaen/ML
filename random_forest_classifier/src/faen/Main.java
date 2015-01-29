package faen;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

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

    private static void readCsv(int m_train, int m_test) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader("data/train.csv"));
        int total = 0;
        // one instance is 28*28 picture.
        instances = new double[m_train][28 * 28];
        targets = new int[m_train];
        test_instances = new double[m_test][28 * 28];
        test_targets = new int[m_test];
        // The first line is header.
        br.readLine();
        while (true) {
            String line = br.readLine();
            if (line == null) {
                break;
            }

            String[] lineArray = line.split(",");
            int m = 28;
            int n = 28;
            int[][] image = new int[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    image[i][j] = getFeatureValue(Integer.parseInt(lineArray[i * m + j + 1]));
                    if (total < m_train) {
                        instances[total][i * m + j] = image[i][j];
                    } else {
                        test_instances[total - m_train][i * m + j] = image[i][j];
                    }

                }
            }
            if (total < m_train) {
                targets[total] = Integer.parseInt(lineArray[0]);
            } else {
                test_targets[total - m_train] = Integer.parseInt(lineArray[0]);
            }
            if (total % 200 == 1) {
                printImage(image);
            }
            if (++total >= m_train + m_test) {
                break;
            }
        }
        br.close();
    }

    private static double[][] instances = null;
    private static int[] targets = null;
    private static double[][] test_instances = null;
    private static int[] test_targets = null;

    public static void main(String[] args) throws Exception {
        readCsv(3000, 500);
        RandomForest rf = new RandomForest();
        rf.train(instances, targets, 100, 40, -1, 1000);
        int correct = 0;
        int wrong = 0;
        for (int i = 0; i < test_targets.length; i++) {
            int actual = rf.predicate(test_instances[i]);
            System.out.println("actual: " + actual + ", expected: " + test_targets[i]);
            if (actual == test_targets[i]) {
                correct++;
            } else {
                wrong++;
            }
        }
        System.out.println("correct:" + correct + ", wrong:" + wrong + ", accuracy:" + 1.0
                * correct / (correct + wrong));
    }
}
