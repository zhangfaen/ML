package yetAnotherImplementationOfBaumWelch;

// Reference: 
//   http://www.itu.dk/people/sestoft/bsa.html
//   http://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
//   https://www.itu.dk/people/sestoft/bsa/Match3.java

// Notational conventions: 

// i     = 1,...,L           indexes x, the observed string, x_0 not a symbol
// k,ell = 0,...,hmm.nstate-1  indexes hmm.state(k)   a_0 is the start state

import java.text.DecimalFormat;
import java.util.Random;

// Some algorithms for Hidden Markov Models : Viterbi,
// Forward, Backward, Baum-Welch.  We compute with log probabilities.

class HMM {
    // State names and state-to-state transition probabilities
    int nstate; // number of states (incl initial state)
    String[] state; // names of the states
    double[][] loga; // loga[k][ell] = log(P(k -> ell))

    // Emission names and emission probabilities
    int nesym; // number of emission symbols
    String esym; // the emission symbols e1,...,eL (characters)
    double[][] loge; // loge[k][ei] = log(P(emit ei in state k))

    // Input:
    // state = array of state names (except initial state)
    // amat = matrix of transition probabilities (except initial state)
    // esym = string of emission names
    // emat = matrix of emission probabilities

    public HMM(String[] state, double[][] amat, double[] pi, String esym, double[][] emat) {
        if (state.length != amat.length)
            throw new IllegalArgumentException("HMM: state and amat disagree");
        if (amat.length != emat.length)
            throw new IllegalArgumentException("HMM: amat and emat disagree");
        for (int i = 0; i < amat.length; i++) {
            if (state.length != amat[i].length)
                throw new IllegalArgumentException("HMM: amat non-square");
            if (esym.length() != emat[i].length)
                throw new IllegalArgumentException("HMM: esym and emat disagree");
        }

        // Set up the transition matrix
        nstate = state.length + 1;
        this.state = new String[nstate];
        loga = new double[nstate][nstate];
        this.state[0] = "B"; // initial state
        // P(start -> start) = 0
        loga[0][0] = Double.NEGATIVE_INFINITY; // = log(0)
        // P(start -> other) = 1.0/state.length
        double fromstart = Math.log(1.0 / state.length);
        for (int j = 1; j < nstate; j++) {
            loga[0][j] = fromstart;
            if (pi != null) {
                loga[0][j] = Math.log(pi[j - 1]);
            }
        }
        for (int i = 1; i < nstate; i++) {
            // Reverse state names for efficient backwards concatenation
            this.state[i] = new StringBuffer(state[i - 1]).reverse().toString();
            // P(other -> start) = 0
            loga[i][0] = Double.NEGATIVE_INFINITY; // = log(0)
            for (int j = 1; j < nstate; j++)
                loga[i][j] = Math.log(amat[i - 1][j - 1]);
        }

        // Set up the emission matrix
        this.esym = esym;
        nesym = esym.length();
        // Assume all esyms are uppercase letters (ASCII <= 91)
        loge = new double[emat.length + 1][91];
        for (int b = 0; b < nesym; b++) {
            // Use the emitted character, not its number, as index into loge:
            char eb = esym.charAt(b);
            // P(emit xi in state 0) = 0
            loge[0][eb] = Double.NEGATIVE_INFINITY; // = log(0)
            for (int k = 0; k < emat.length; k++)
                loge[k + 1][eb] = Math.log(emat[k][b]);
        }
    }

    public void print(SystemOut out) {
        printpi(out);
        printa(out);
        printe(out);
    }

    public void printpi(SystemOut out) {
        out.println("Start probabilities:");
        for (int i = 1; i < nstate; i++) {
            out.print(fmtlog(loga[0][i]));
        }
        out.println();
    }

    public void printa(SystemOut out) {
        out.println("Transition probabilities:");
        for (int i = 1; i < nstate; i++) {
            for (int j = 1; j < nstate; j++)
                out.print(fmtlog(loga[i][j]));
            out.println();
        }
    }

    public void printe(SystemOut out) {
        out.println("Emission probabilities:");
        for (int b = 0; b < nesym; b++)
            out.print(esym.charAt(b) + "        ");
        out.println();
        for (int i = 1; i < loge.length; i++) {
            for (int b = 0; b < nesym; b++)
                out.print(fmtlog(loge[i][esym.charAt(b)]));
            out.println();
        }
    }

    private static DecimalFormat fmt = new DecimalFormat("0.000000 ");

    public static String fmtlog(double x) {
        if (x == Double.NEGATIVE_INFINITY)
            return fmt.format(0);
        else
            return fmt.format(Math.exp(x));
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

    // The Baum-Welch algorithm for estimating HMM parameters for a
    // given model topology and a family of observed sequences.
    // Often gets stuck at a non-global minimum; depends on initial guess.

    // xs is the set of training sequences
    // state is the set of HMM state names
    // esym is the set of emissible symbols
    public static HMM baumwelch(String[] xs, String[] state, String esym, final double threshold) {
        int nstate = state.length;
        int nseqs = xs.length;
        int nesym = esym.length();

        Forward[] fwds = new Forward[nseqs];
        Backward[] bwds = new Backward[nseqs];
        double[] logP = new double[nseqs];

        double[][] amat = new double[nstate][];
        double[][] emat = new double[nstate][];
        double[] pi = new double[nstate];

        // Set up the inverse of b -> esym.charAt(b); assume all esyms <= 'Z'
        int[] esyminv = new int[91];
        for (int i = 0; i < esyminv.length; i++)
            esyminv[i] = -1;
        for (int b = 0; b < nesym; b++)
            esyminv[esym.charAt(b)] = b;

        // Initially use random transition and emission matrices
        for (int k = 0; k < nstate; k++) {
            amat[k] = randomdiscrete(nstate);
            emat[k] = randomdiscrete(nesym);
        }
        pi = randomdiscrete(nstate);

        HMM hmm = new HMM(state, amat, pi, esym, emat);

        double oldloglikelihood;

        // Compute Forward and Backward tables for the sequences
        double loglikelihood = fwdbwd(hmm, xs, fwds, bwds, logP);
        System.out.println("log likelihood = " + loglikelihood);
        // hmm.print(new SystemOut());
        do {
            oldloglikelihood = loglikelihood;
            // Compute estimates for A and E
            double[][] A = new double[nstate][nstate];
            double[][] E = new double[nstate][nesym];
            double[] PI = new double[nstate];
            for (int s = 0; s < nseqs; s++) {
                String x = xs[s];
                Forward fwd = fwds[s];
                Backward bwd = bwds[s];
                int L = x.length();
                double P = logP[s];
                for (int i = 0; i < L; i++) {
                    for (int k = 0; k < nstate; k++)
                        E[k][esyminv[x.charAt(i)]] += exp(fwd.f[i + 1][k + 1] + bwd.b[i + 1][k + 1]
                                - P);
                }
                for (int i = 0; i < L - 1; i++)
                    for (int k = 0; k < nstate; k++)
                        for (int ell = 0; ell < nstate; ell++)
                            A[k][ell] += exp(fwd.f[i + 1][k + 1] + hmm.loga[k + 1][ell + 1]
                                    + hmm.loge[ell + 1][x.charAt(i + 1)] + bwd.b[i + 2][ell + 1]
                                    - P);
                double pisum = 0;
                for (int k = 0; k < nstate; k++) {
                    pisum += exp(fwd.f[1][k + 1] + bwd.b[1][k + 1]);
                }
                for (int k = 0; k < nstate; k++) {
                    PI[k] += exp(fwd.f[1][k + 1] + bwd.b[1][k + 1]) / pisum;
                }
            }
            // Estimate new model parameters
            normalize(PI);
            for (int k = 0; k < nstate; k++) {
                double Aksum = 0;
                for (int ell = 0; ell < nstate; ell++)
                    Aksum += A[k][ell];
                for (int ell = 0; ell < nstate; ell++)
                    amat[k][ell] = A[k][ell] / Aksum;
                double Eksum = 0;
                for (int b = 0; b < nesym; b++)
                    Eksum += E[k][b];
                for (int b = 0; b < nesym; b++)
                    emat[k][b] = E[k][b] / Eksum;
                pi[k] = PI[k];
            }
            // Create new model
            hmm = new HMM(state, amat, pi, esym, emat);
            loglikelihood = fwdbwd(hmm, xs, fwds, bwds, logP);
            System.out.println("log likelihood = " + loglikelihood);
            // hmm.print(new SystemOut());
        } while (Math.abs(oldloglikelihood - loglikelihood) > threshold);
        return hmm;
    }

    private static double fwdbwd(HMM hmm, String[] xs, Forward[] fwds, Backward[] bwds,
            double[] logP) {
        double loglikelihood = 0;
        for (int s = 0; s < xs.length; s++) {
            fwds[s] = new Forward(hmm, xs[s]);
            bwds[s] = new Backward(hmm, xs[s]);
            logP[s] = fwds[s].logprob();
            loglikelihood += logP[s];
        }
        return loglikelihood;
    }

    public static double exp(double x) {
        if (x == Double.NEGATIVE_INFINITY)
            return 0;
        else
            return Math.exp(x);
    }

    private static double[] randomdiscrete(int n) {
        double[] ps = new double[n];
        double sum = 0;
        // Generate random numbers
        for (int i = 0; i < n; i++) {
            ps[i] = Math.random();
            sum += ps[i];
        }
        // Scale to obtain a discrete probability distribution
        for (int i = 0; i < n; i++)
            ps[i] /= sum;
        return ps;
    }
}

abstract class HMMAlgo {
    HMM hmm; // the hidden Markov model
    String x; // the observed string of emissions

    public HMMAlgo(HMM hmm, String x) {
        this.hmm = hmm;
        this.x = x;
    }

    // Compute log(p+q) from plog = log p and qlog = log q, using that
    // log (p + q) = log (p(1 + q/p)) = log p + log(1 + q/p)
    // = log p + log(1 + exp(log q - log p)) = plog + log(1 + exp(logq - logp))
    // and that log(1 + exp(d)) < 1E-17 for d < -37.

    static double logplus(double plog, double qlog) {
        double max, diff;
        if (plog > qlog) {
            if (qlog == Double.NEGATIVE_INFINITY)
                return plog;
            else {
                max = plog;
                diff = qlog - plog;
            }
        } else {
            if (plog == Double.NEGATIVE_INFINITY)
                return qlog;
            else {
                max = qlog;
                diff = plog - qlog;
            }
        }
        // Now diff <= 0 so Math.exp(diff) will not overflow
        return max + (diff < -37 ? 0 : Math.log(1 + Math.exp(diff)));
    }
}

class TestLogPlus {
    // Test HMMAlgo.logplus: it passes these tests
    public static void main(String[] args) {
        final double EPS = 1E-14;
        Random rnd = new Random();
        int count = Integer.parseInt(args[0]);
        for (int k = 200; k >= -200; k--)
            for (int i = 0; i < count; i++) {
                double logp = Math.abs(rnd.nextDouble()) * Math.pow(10, k);
                double logq = Math.abs(rnd.nextDouble());
                double logpplusq = HMMAlgo.logplus(logp, logq);
                double p = Math.exp(logp), q = Math.exp(logq), pplusq = Math.exp(logpplusq);
                if (Math.abs(p + q - pplusq) > EPS * pplusq)
                    System.out.println(p + "+" + q + "-" + pplusq);
            }
    }
}

// The Viterbi algorithm: find the most probable state path producing
// the observed outputs x

class Viterbi extends HMMAlgo {
    double[][] v; // the matrix used to find the decoding
                  // v[i][k] = v_k(i) =
                  // log(max(P(pi in state k has sym i | path pi)))
    Traceback2[][] B; // the traceback matrix
    Traceback2 B0; // the start of the traceback

    public Viterbi(HMM hmm, String x) {
        super(hmm, x);
        final int L = x.length();
        v = new double[L + 1][hmm.nstate];
        B = new Traceback2[L + 1][hmm.nstate];
        v[0][0] = 0; // = log(1)
        for (int k = 1; k < hmm.nstate; k++)
            v[0][k] = Double.NEGATIVE_INFINITY; // = log(0)
        for (int i = 1; i <= L; i++)
            v[i][0] = Double.NEGATIVE_INFINITY; // = log(0)
        for (int i = 1; i <= L; i++)
            for (int ell = 0; ell < hmm.nstate; ell++) {
                int kmax = 0;
                double maxprod = v[i - 1][kmax] + hmm.loga[kmax][ell];
                for (int k = 1; k < hmm.nstate; k++) {
                    double prod = v[i - 1][k] + hmm.loga[k][ell];
                    if (prod > maxprod) {
                        kmax = k;
                        maxprod = prod;
                    }
                }
                v[i][ell] = hmm.loge[ell][x.charAt(i - 1)] + maxprod;
                B[i][ell] = new Traceback2(i - 1, kmax);
            }
        int kmax = 0;
        double max = v[L][kmax];
        for (int k = 1; k < hmm.nstate; k++) {
            if (v[L][k] > max) {
                kmax = k;
                max = v[L][k];
            }
        }
        B0 = new Traceback2(L, kmax);
    }

    public String getPath() {
        StringBuffer res = new StringBuffer();
        Traceback2 tb = B0;
        int i = tb.i, j = tb.j;
        while ((tb = B[tb.i][tb.j]) != null) {
            res.append(hmm.state[j]);
            i = tb.i;
            j = tb.j;
        }
        return res.reverse().toString();
    }

    public void print(SystemOut out) {
        for (int j = 0; j < hmm.nstate; j++) {
            for (int i = 0; i < v.length; i++)
                out.print(HMM.fmtlog(v[i][j]));
            out.println();
        }
    }
}

// The `Forward algorithm': find the probability of an observed sequence x
class Forward extends HMMAlgo {
    double[][] f; // the matrix used to find the decoding
                  // f[i][k] = f_k(i) = log(P(x1..xi, pi_i=k))
    private int L;

    public Forward(HMM hmm, String x) {
        super(hmm, x);
        L = x.length();
        f = new double[L + 1][hmm.nstate];
        f[0][0] = 0; // = log(1)
        for (int k = 1; k < hmm.nstate; k++)
            f[0][k] = Double.NEGATIVE_INFINITY; // = log(0)
        for (int i = 1; i <= L; i++)
            f[i][0] = Double.NEGATIVE_INFINITY; // = log(0)
        for (int i = 1; i <= L; i++)
            for (int ell = 1; ell < hmm.nstate; ell++) {
                double sum = Double.NEGATIVE_INFINITY; // = log(0)
                for (int k = 0; k < hmm.nstate; k++)
                    sum = logplus(sum, f[i - 1][k] + hmm.loga[k][ell]);
                f[i][ell] = hmm.loge[ell][x.charAt(i - 1)] + sum;
            }
    }

    double logprob() {
        double sum = Double.NEGATIVE_INFINITY; // = log(0)
        for (int k = 0; k < hmm.nstate; k++)
            sum = logplus(sum, f[L][k]);
        return sum;
    }

    public void print(SystemOut out) {
        for (int j = 0; j < hmm.nstate; j++) {
            for (int i = 0; i < f.length; i++)
                out.print(HMM.fmtlog(f[i][j]));
            out.println();
        }
    }
}

// The `Backward algorithm': find the probability of an observed sequence x

class Backward extends HMMAlgo {
    double[][] b; // the matrix used to find the decoding
                  // b[i][k] = b_k(i) = log(P(x(i+1)..xL, pi_i=k))

    public Backward(HMM hmm, String x) {
        super(hmm, x);
        int L = x.length();
        b = new double[L + 1][hmm.nstate];
        for (int k = 1; k < hmm.nstate; k++)
            b[L][k] = 0; // = log(1) // should be hmm.loga[k][0]
        for (int i = L - 1; i >= 1; i--)
            for (int k = 0; k < hmm.nstate; k++) {
                double sum = Double.NEGATIVE_INFINITY; // = log(0)
                for (int ell = 1; ell < hmm.nstate; ell++)
                    sum = logplus(sum, hmm.loga[k][ell] + hmm.loge[ell][x.charAt(i)]
                            + b[i + 1][ell]);
                b[i][k] = sum;
            }
    }

    double logprob() {
        double sum = Double.NEGATIVE_INFINITY; // = log(0)
        for (int ell = 0; ell < hmm.nstate; ell++)
            sum = logplus(sum, hmm.loga[0][ell] + hmm.loge[ell][x.charAt(0)] + b[1][ell]);
        return sum;
    }

    public void print(SystemOut out) {
        for (int j = 0; j < hmm.nstate; j++) {
            for (int i = 0; i < b.length; i++)
                out.print(HMM.fmtlog(b[i][j]));
            out.println();
        }
    }
}

// Compute posterior probabilities using Forward and Backward

class PosteriorProb {
    Forward fwd; // result of the forward algorithm
    Backward bwd; // result of the backward algorithm
    private double logprob;

    PosteriorProb(Forward fwd, Backward bwd) {
        this.fwd = fwd;
        this.bwd = bwd;
        logprob = fwd.logprob(); // should equal bwd.logprob()
    }

    double posterior(int i, int k) // i=index into the seq; k=the HMM state
    {
        return Math.exp(fwd.f[i][k] + bwd.b[i][k] - logprob);
    }
}

// Traceback objects

abstract class Traceback {
    int i, j; // absolute coordinates
}

// Traceback2 objects

class Traceback2 extends Traceback {
    public Traceback2(int i, int j) {
        this.i = i;
        this.j = j;
    }
}

// Auxiliary classes for output

class SystemOut {
    public void print(String s) {
        System.out.print(s);
    }

    public void println(String s) {
        System.out.println(s);
    }

    public void println() {
        System.out.println();
    }
}

public class HmmStuff {

}