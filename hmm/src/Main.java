import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Main {
    public static int chooseOneRandomly(double[] distribution) {
        double rand = Math.random();
        double acc = 0;
        for (int i = 0; i < distribution.length; i++) {
            acc += distribution[i];
            if (rand <= acc) {
                return i;
            }
        }
        // Must not be here.
        return -1;
    }

    public static void main(String[] args) throws IOException {
        // Two dices A and B
        double[] stateProb = { 0.3, 0.7 };
        double[][] stateTrans = { { 0.1, 0.9 }, { 0.2, 0.8 } };
        double[][] emission = { { 0.2, 0.8 }, { 0.9, 0.1 } };

        BufferedWriter bw = new BufferedWriter(new FileWriter(
                "/Users/zhangfaen/dev/ml/kaggle/hmm/data/data.txt"));
        int samples = 1000;
        for (int i = 0; i < samples; i++) {
            int diceIndex = chooseOneRandomly(stateProb);
            for (int j = 0; j < 200; j++) {
                int output = chooseOneRandomly(emission[diceIndex]);
                bw.write("" + output + ",");
                diceIndex = chooseOneRandomly(stateTrans[diceIndex]);
            }
            bw.write("\n");
        }
        bw.close();

        HmmBaumWelch hbw = new HmmBaumWelch();
        hbw.initParam(2, "/Users/zhangfaen/dev/ml/kaggle/hmm/data/data.txt");
        // hbw.initParam(2, "/Users/zhangfaen/dev/ml/kaggle/hmm/data/test.txt");
        // hbw.setSpecificInitalState(new double[] { 0.85, 0.15 }, new
        // double[][] { { 0.3, 0.7 },
        // { 0.1, 0.9 } }, new double[][] { { 0.4, 0.6 }, { 0.5, 0.5 } });

        hbw.print();
        hbw.baumWelch(5000);
        hbw.print();
        hbw.setSpecificInitalState(stateProb, stateTrans, emission);
        hbw.baumWelch(1);
        hbw.print();
    }
}
