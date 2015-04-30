import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Main {

    static Random rand = new Random();
    
    
    static int[][] C = new int[20][20];

    static void initC() {
        C[0][0] = 1;
        for (int i = 0; i < C.length; i++) {
            for (int j = 0; j < C[0].length; j++) {
                if (i == j) {
                    C[i][j] = 1;
                } else if (i > j) {
                    if (j == 0) {
                        C[i][j] = 1;
                    } else {
                        C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
                    }
                }
                // System.out.println("C[" + i + "][" + j + "] = " + C[i][j]);
            }
        }
    }

    public static void main(String[] args) {
        initC();
        int numberOfCoins = 3;
        int numberOfExperiments = 10000;
        int numberOfFlips = 15;
        double[] ps = new double[numberOfCoins];
        ps[0] = 0.2;
        ps[1] = 0.5;
        ps[2] = 0.8;

        List<Integer> experiments = new ArrayList<Integer>();
        for (int i = 0; i < numberOfExperiments; i++) {
            int heads = 0;
            // First choose a coin randomly
            int index = rand.nextInt(numberOfCoins);
            for (int j = 0; j < numberOfFlips; j++) {
                if (rand.nextDouble() <= ps[index]) {
                    heads++;
                }
            }
            experiments.add(heads);
        }
        System.out.println(experiments.toString());

        // Now estimate parameters by EM algorithm

        double[] eps = new double[numberOfCoins];
        for (int k = 0; k < numberOfCoins; k++) {
            eps[k] = rand.nextDouble();
        }
        for (int i = 0; i < 100; i++) {
            double[][] posts = new double[numberOfExperiments][numberOfCoins];
            for (int j = 0; j < numberOfExperiments; j++) {
                double total = 0;
                for (int k = 0; k < numberOfCoins; k++) {
                    double curEp = eps[k];
                    int heads = experiments.get(j);
                    double curPost = C[numberOfFlips][heads] * Math.pow(curEp, heads)
                            * Math.pow(1 - curEp, numberOfFlips - heads);
                    total += curPost;
                    posts[j][k] = curPost;
                }
                for (int k = 0; k < numberOfCoins; k++) {
                    posts[j][k] /= total;
                }

            }
            for (int k = 0; k < numberOfCoins; k++) {
                double heads = 0;
                double tails = 0;
                for (int j = 0; j < numberOfExperiments; j++) {
                    heads += posts[j][k] * experiments.get(j);
                    tails += posts[j][k] * (numberOfFlips - experiments.get(j));
                }
                eps[k] = heads / (heads + tails);
            }
        }

        System.out.println(Arrays.toString(eps));

    }
}
