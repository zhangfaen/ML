package faen;

import java.util.Arrays;
import java.util.Random;

// http://en.wikipedia.org/wiki/Artificial_neural_network
// A kind of non linear model of Machine Learning.
public class NN {
    static class Util {
        public static void CHECK(boolean condition, String message) {
            if (!condition) {
                throw new RuntimeException(message);
            }
        }
    }

    private int expandedInputNodes;
    private int hiddenNodes;
    private int outputNodes;

    // Weights matrix between input layer and hidden layer
    private double[][] wi;
    // Weights matrix between hidden layer and output layer.
    private double[][] wo;

    // last change in weights for momentum
    private double[][] wi_momentum;
    // last change in weights for momentum
    private double[][] wo_momentum;

    // Expanded instance, whose size is this.outputSize + 1.
    // The last element will be fixed to 1.0
    private double[] expandedInstance;

    private double[] hiddenActivations;
    private double[] outputActivations;

    // The sigmoid function: s(x) = 1 / (1 + (e^-x))
    // The derivative of s(x): s(x) * (1 - s(x))
    private double s(double x) {
        return 1.0 / (1.0 + Math.pow(Math.E, -x));
    }

    public NN(int featuresOfInstance, int nodesOfHiddenLayer, int nodesOfOutputLayer) {
        Util.CHECK(featuresOfInstance > 0, "");
        Util.CHECK(nodesOfHiddenLayer > 0, "");
        Util.CHECK(nodesOfOutputLayer > 0, "");
        this.expandedInputNodes = featuresOfInstance + 1;
        this.hiddenNodes = nodesOfHiddenLayer;
        this.outputNodes = nodesOfOutputLayer;
        this.wi = new double[this.expandedInputNodes][this.hiddenNodes];
        this.wo = new double[this.hiddenNodes][this.outputNodes];
        this.wi_momentum = new double[this.expandedInputNodes][this.hiddenNodes];
        this.wo_momentum = new double[this.hiddenNodes][this.outputNodes];
        this.expandedInstance = new double[this.expandedInputNodes];
        this.expandedInstance[this.expandedInputNodes - 1] = 1.0;
        this.hiddenActivations = new double[this.hiddenNodes];
        this.outputActivations = new double[this.outputNodes];
    }

    // Randomly initialize the input and output weights matrix
    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < this.wi.length; i++) {
            for (int j = 0; j < this.wi[0].length; j++) {
                // [-2.0, 2.0]
                this.wi[i][j] = rand.nextDouble() * 4 - 2;
            }
        }
        for (int i = 0; i < this.wo.length; i++) {
            for (int j = 0; j < this.wo[0].length; j++) {
                // [-2.0, 2.0]
                this.wo[i][j] = rand.nextDouble() * 4 - 2;
            }
        }
    }

    private void forwardPropagate(double[] instance) {
        Util.CHECK(instance.length + 1 == this.expandedInputNodes, "");
        for (int i = 0; i < instance.length; i++) {
            // Note: the last element of this.expandedInstance will be 1.0
            this.expandedInstance[i] = instance[i];
        }

        // forward propagation
        for (int j = 0; j < this.hiddenNodes; j++) {
            double tmp = 0;
            for (int i = 0; i < this.expandedInputNodes; i++) {
                tmp += this.expandedInstance[i] * this.wi[i][j];
            }
            this.hiddenActivations[j] = s(tmp);
        }
        for (int k = 0; k < this.outputNodes; k++) {
            double tmp = 0;
            for (int j = 0; j < this.hiddenNodes; j++) {
                tmp += this.hiddenActivations[j] * this.wo[j][k];
            }
            this.outputActivations[k] = s(tmp);
        }
    }

    // Predicate output from one instance.
    public double[] predicate(double[] instance) {
        forwardPropagate(instance);
        return this.outputActivations.clone();
    }

    // Update NN weights by one instance and its label.
    private double feedOneInstance(double[] instance, double[] target, double rate, double momentum) {
        forwardPropagate(instance);
        double error = 0;
        for (int k = 0; k < this.outputNodes; k++) {
            error += 0.5 * (target[k] - this.outputActivations[k])
                    * (target[k] - this.outputActivations[k]);
        }

        // backward propagation
        // update output weights matrix
        for (int j = 0; j < this.hiddenNodes; j++) {
            for (int k = 0; k < this.outputNodes; k++) {
                // wo[j,k]
                double change = (this.outputActivations[k] - target[k]) * this.outputActivations[k]
                        * (1 - this.outputActivations[k]);
                change *= this.hiddenActivations[j];
                this.wo[j][k] = this.wo[j][k] - rate * change - momentum * this.wo_momentum[j][k];
                this.wo_momentum[j][k] = change;
            }
        }
        // update input weights matrix
        for (int i = 0; i < this.expandedInputNodes; i++) {
            for (int j = 0; j < this.hiddenNodes; j++) {
                // wi[i, j]
                double change = 0;
                for (int k = 0; k < this.outputNodes; k++) {
                    change += (this.outputActivations[k] - target[k]) * this.outputActivations[k]
                            * (1 - this.outputActivations[k]) * this.wo[j][k];

                }
                change *= this.hiddenActivations[j] * (1 - this.hiddenActivations[j]);
                change *= this.expandedInstance[i];
                this.wi[i][j] = this.wi[i][j] - rate * change - momentum * this.wi_momentum[i][j];
                this.wi_momentum[i][j] = change;
            }
        }
        return error;
    }

    // Train the NN
    public void train(double[][] instances, double[][] targets, int iterations, double rate,
            double momentum) {
        Util.CHECK(instances.length == targets.length && targets.length > 0, "");
        Util.CHECK(instances[0].length > 0, "");
        Util.CHECK(targets[0].length == this.outputNodes, "");
        initializeWeights();
        for (int it = 0; it < iterations; it++) {
            double error = 0;
            for (int index = 0; index < instances.length; index++) {
                double[] instance = instances[index];
                double[] target = targets[index];
                error += feedOneInstance(instance, target, rate, momentum);
            }
            if (it % 20 == 0) {
                System.out.println("error: " + error);
            }
        }
    }

    // Bits XOR
    private static void demo1() {
        double[][] instances = new double[][] { { 1, 0 }, { 1, 1 }, { 0, 1 }, { 0, 0 } };
        double[][] targets = new double[][] { { 1 }, { 0 }, { 1 }, { 0 } };
        NN nn = new NN(2, 4, 1);
        nn.train(instances, targets, 10000, 1.5, 0.2);
        System.out.println("1 xor 1: " + Arrays.toString(nn.predicate(new double[] { 1, 1 })));
        System.out.println("1 xor 0: " + Arrays.toString(nn.predicate(new double[] { 1, 0 })));
        System.out.println("0 xor 0: " + Arrays.toString(nn.predicate(new double[] { 0, 0 })));
        System.out.println("0 xor 1: " + Arrays.toString(nn.predicate(new double[] { 0, 1 })));
    }

    // Data points are along 2 circles: x^2 + y^2 = 2 and x^2 + y^2 = 4
    // The data points are along the first circle have label 0.
    // The data points are along the second circle have label 1.
    // Note: we should not use labels 2 and 4, because the output is an
    // activation function,
    // "> 0" means activated.
    private static void demo2() {
        int m = 0;
        double step = 0.5;
        for (double i = 0; i < Math.PI * 2; i += step) {
            m++;
        }
        double[][] instances = new double[m * 2][2];
        double[][] targets = new double[m * 2][1];
        int index = 0;
        for (double i = 0; i < Math.PI * 2; i += step) {

            instances[index][0] = 2 * Math.cos(i);
            instances[index][1] = 2 * Math.sin(i);
            instances[index + m][0] = 4 * Math.cos(i);
            instances[index + m][1] = 4 * Math.sin(i);
            targets[index][0] = 0;
            targets[index + m][0] = 1;
            index++;
        }
        NN nn = new NN(2, 100, 1);
        nn.train(instances, targets, 5000, 0.5, 0.2);

        // Testing.
        for (double i = 0.2; i < Math.PI * 2; i += 1) {
            double x = 2 * Math.cos(i);
            double y = 2 * Math.sin(i);
            System.out.println("x: " + x + " y:" + y + " r:" + (Math.hypot(x, y)) + " result:"
                    + Arrays.toString(nn.predicate(new double[] { x, y })));
            x = 4 * Math.cos(i);
            y = 4 * Math.sin(i);
            System.out.println("x: " + x + " y:" + y + " r:" + (Math.hypot(x, y)) + " result:"
                    + Arrays.toString(nn.predicate(new double[] { x, y })));

        }
    }

    public static void main(String[] args) {
        demo1();
        System.out.println("-----------------------------------------");
        demo2();
    }
}
