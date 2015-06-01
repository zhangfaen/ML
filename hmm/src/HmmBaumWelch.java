import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Learn parameters of HMM model by Baum-Welch algorithm, see
 * http://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
 */
public class HmmBaumWelch {

    // Number of hidden states
    private int hiddenStateCount;

    private int visiableStateCount;

    // Initial state distribution
    private double[] stateProb;
    // Probability matrix from one hidden state to another hidden state.
    private double[][] stateTrans;
    // Probability matrix from one hidden state to one observable state
    private double[][] emission;

    // All observed sequences.
    private List<int[]> observeSeqs = new ArrayList<int[]>();

    /**
     * When to stop iteration of EM algorithm.
     */
    private final double DELTA_PI = 1E-7;
    private final double DELTA_A = 1E-7;
    private final double DELTA_B = 1E-7;

    /**
     * 
     * @param stateCount
     *            specify number of hidden states
     * @param observeFile
     *            file containing sequences to train the model. each line is a
     *            sequence, each observable state in the sequence is separated
     *            by a comma (',')
     * @throws IOException
     */
    public void initParam(int hiddenStateCount, String observeFile) throws IOException {
        this.hiddenStateCount = hiddenStateCount;
        BufferedReader br = new BufferedReader(new FileReader(new File(observeFile)));
        String line = null;
        Set<String> allVisiableStates = new HashSet<String>();
        while ((line = br.readLine()) != null) {
            String[] arr = line.split(",");
            if (arr.length <= 1) {
                // Skip the sequences whose length is <= 1.
                continue;
            }
            int[] seq = new int[arr.length];
            for (int i = 0; i < arr.length; i++) {
                String observe = arr[i];
                seq[i] = Integer.parseInt(observe);
                allVisiableStates.add(observe);
            }
            observeSeqs.add(seq);
        }
        br.close();
        this.visiableStateCount = allVisiableStates.size();

        stateProb = new double[hiddenStateCount];
        initWeightRandomly(stateProb);
        stateTrans = new double[hiddenStateCount][];
        for (int i = 0; i < hiddenStateCount; i++) {
            stateTrans[i] = new double[hiddenStateCount];
            initWeightRandomly(stateTrans[i]);
        }
        emission = new double[hiddenStateCount][];
        for (int i = 0; i < hiddenStateCount; i++) {
            emission[i] = new double[visiableStateCount];
            initWeightRandomly(emission[i]);
        }
    }

    public void setSpecificInitalState(double[] stateProb, double[][] stateTrans,
            double[][] emission) {
        this.stateProb = stateProb;
        this.stateTrans = stateTrans;
        this.emission = emission;
    }

    /**
     * Initialize arr with random positive weights and make sure arr is
     * normalized.
     * 
     * @param arr
     */
    public void initWeightRandomly(double[] arr) {
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            arr[i] = 0.5 + Math.random();
            sum += arr[i];
        }
        for (int i = 0; i < arr.length; i++) {
            arr[i] /= sum;
        }
    }

    /**
     * BaumWelch algorithm.
     */
    public void baumWelch(int iterations) {
        long begin = System.currentTimeMillis();
        int iter = 0;
        while (iter++ < iterations) {
            double[] stateProb_new = new double[hiddenStateCount];
            double[][] stateTrans_new = new double[hiddenStateCount][];
            double[][] emission_new = new double[hiddenStateCount][];
            for (int i = 0; i < hiddenStateCount; i++) {
                stateTrans_new[i] = new double[hiddenStateCount];
            }
            for (int i = 0; i < hiddenStateCount; i++) {
                emission_new[i] = new double[visiableStateCount];
            }
            double logLikelihood = 0;
            for (int[] seq : observeSeqs) {
                int T = seq.length;
                double[][] alpha = new double[T][];
                double[][] beta = new double[T][];
                double[][] gamma = new double[T][];
                for (int i = 0; i < T; i++) {
                    alpha[i] = new double[hiddenStateCount];
                    beta[i] = new double[hiddenStateCount];
                    gamma[i] = new double[hiddenStateCount];
                }
                double[][][] xi = new double[T - 1][][];
                for (int i = 0; i < T - 1; i++) {
                    xi[i] = new double[hiddenStateCount][];
                    for (int j = 0; j < hiddenStateCount; j++) {
                        xi[i][j] = new double[hiddenStateCount];
                    }
                }
                int observeIndex = seq[0];
                // compute alpha
                for (int i = 0; i < hiddenStateCount; i++) {
                    alpha[0][i] = stateProb[i] * emission[i][observeIndex];
                }
                for (int t = 1; t < T; t++) {
                    observeIndex = seq[t];
                    for (int j = 0; j < hiddenStateCount; j++) {
                        double sum = 0;
                        for (int i = 0; i < hiddenStateCount; i++) {
                            sum += alpha[t - 1][i] * stateTrans[i][j];
                        }
                        alpha[t][j] = sum * emission[j][observeIndex];
                    }
                }
                // compute beta
                for (int i = 0; i < hiddenStateCount; i++) {
                    beta[T - 1][i] = 1;
                }
                for (int t = T - 2; t >= 0; t--) {
                    observeIndex = seq[t + 1];
                    for (int i = 0; i < hiddenStateCount; i++) {
                        double sum = 0;
                        for (int j = 0; j < hiddenStateCount; j++) {
                            sum += beta[t + 1][j] * stateTrans[i][j] * emission[j][observeIndex];
                        }
                        beta[t][i] = sum;
                    }
                }
                double[] denominator = new double[T];
                // compute gamma (a.k.a γ)
                for (int t = 0; t < T; t++) {
                    double sum = 0;
                    for (int j = 0; j < hiddenStateCount; j++) {
                        sum += alpha[t][j] * beta[t][j];
                    }
                    denominator[t] = sum;
                    for (int i = 0; i < hiddenStateCount; i++) {
                        gamma[t][i] = alpha[t][i] * beta[t][i] / sum;
                    }
                }
                // compute xi (a.k.a ξ)
                for (int t = 0; t < T - 1; t++) {
                    observeIndex = seq[t + 1];
                    for (int i = 0; i < hiddenStateCount; i++) {
                        for (int j = 0; j < hiddenStateCount; j++) {
                            xi[t][i][j] = alpha[t][i] * stateTrans[i][j] * beta[t + 1][j]
                                    * emission[j][observeIndex] / denominator[t];
                        }
                    }
                }
                // compute stateProb
                double[] curr_stateProb = new double[hiddenStateCount];
                for (int i = 0; i < hiddenStateCount; i++) {
                    curr_stateProb[i] = gamma[0][i];
                    stateProb_new[i] += curr_stateProb[i];
                }
                // compute stateTrans
                double[][] curr_stateTrans = new double[hiddenStateCount][];
                for (int i = 0; i < hiddenStateCount; i++) {
                    curr_stateTrans[i] = new double[hiddenStateCount];
                    for (int j = 0; j < hiddenStateCount; j++) {
                        double up = 0;
                        double down = 0;
                        for (int t = 0; t < T - 1; t++) {
                            up += xi[t][i][j];
                            down += gamma[t][i];
                        }
                        if (down > 0) {
                            // Make sure we won't get a NAN
                            curr_stateTrans[i][j] = up / down;
                            stateTrans_new[i][j] += curr_stateTrans[i][j];
                        } else {
                            // If down is 0, we use the old value.
                            stateTrans_new[i][j] += stateTrans[i][j];
                            System.err.println("up=" + up + ",down=" + down);
                        }
                    }
                }
                // compute emission
                double[][] curr_emission = new double[hiddenStateCount][];
                for (int i = 0; i < hiddenStateCount; i++) {
                    curr_emission[i] = new double[this.visiableStateCount];
                    for (int j = 0; j < this.visiableStateCount; j++) {
                        double up = 0;
                        double down = 0;
                        for (int t = 0; t < T; t++) {
                            if (j == seq[t]) {
                                up += gamma[t][i];
                            }
                            down += gamma[t][i];
                        }
                        curr_emission[i][j] = up / down;
                        emission_new[i][j] += curr_emission[i][j];
                    }
                }
                double seqProb = 0;
                for (int i = 0; i < hiddenStateCount; i++) {
                    seqProb += alpha[seq.length - 1][i];
                }
                logLikelihood += Math.log(seqProb);
            }

            // update parameters in batch
            double delta_pi = 0;
            double delta_a = 0;
            double delta_b = 0;
            int seqCount = observeSeqs.size();

            normalize(stateProb_new);
            for (int i = 0; i < hiddenStateCount; i++) {
                // stateProb_new[i] /= seqCount;
                delta_pi += Math.abs(stateProb_new[i] - stateProb[i]);
            }
            for (int i = 0; i < hiddenStateCount; i++) {
                normalize(stateTrans_new[i]);
                for (int j = 0; j < hiddenStateCount; j++) {
                    // stateTrans_new[i][j] /= seqCount;
                    delta_a += Math.abs(stateTrans_new[i][j] - stateTrans[i][j]);
                }
            }
            for (int i = 0; i < hiddenStateCount; i++) {
                normalize(emission_new[i]);
                for (int j = 0; j < this.visiableStateCount; j++) {
                    // emission_new[i][j] /= seqCount;
                    delta_b += Math.abs(emission_new[i][j] - emission[i][j]);
                }
            }
            System.out.println("iteration " + iter + ", delta_pi=" + delta_pi + ", delta_a="
                    + delta_a + ", delta_b=" + delta_b + ", logLikelihood:" + logLikelihood);
            if (delta_pi <= DELTA_PI && delta_a <= DELTA_A && delta_b <= DELTA_B) {
                break;
            } else {
                stateProb = stateProb_new;
                stateTrans = stateTrans_new;
                emission = emission_new;
            }
        }
        long end = System.currentTimeMillis();
        System.out.println("time elapse " + (end - begin) / 1000 + " seconds.");
    }

    static void normalize(double[] all) {
        double sum = 0;
        for (double d : all) {
            sum += d;
        }
        for (int i = 0; i < all.length; i++) {
            all[i] /= sum;
        }
    }

    public int getStateCount() {
        return hiddenStateCount;
    }

    public double[] getStateProb() {
        return stateProb;
    }

    public double[][] getStateTrans() {
        return stateTrans;
    }

    public double[][] getEmission() {
        return emission;
    }

    public void printStateProb() {
        System.out.print("state prob:");
        for (int i = 0; i < stateProb.length; i++) {
            System.out.print(String.format("%.3f", stateProb[i]) + ",");
        }
        System.out.println();
    }

    public void printStateTrans() {
        System.out.println("state trans:");
        for (int i = 0; i < stateTrans.length; i++) {
            for (int j = 0; j < stateTrans[i].length; j++) {
                System.out.print(String.format("%.3f", stateTrans[i][j]) + ",");
            }
            System.out.println();
        }
    }

    public void printStateEmission() {
        System.out.println("emissions:");
        for (int i = 0; i < emission.length; i++) {
            for (int j = 0; j < emission[i].length; j++) {
                System.out.print(String.format("%.3f", emission[i][j]) + ",");
            }
            System.out.println();
        }
    }

    public void print() {
        printStateProb();
        printStateTrans();
        printStateEmission();
    }
}