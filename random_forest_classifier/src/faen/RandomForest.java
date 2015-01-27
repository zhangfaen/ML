package faen;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class RandomForest {
    static class Util {
        public static void CHECK(boolean condition, String message) {
            if (!condition) {
                throw new RuntimeException(message);
            }
        }
    }

    private double[][] instances_;
    private int[] targets_;
    private int numOfTrees_;
    private int numOfFeatures_;
    private int maxDepth_;
    private TreeNode[] trees_;

    /**
     * Train the RF model
     * 
     * @param instances
     * @param targets
     * @param numOfTrees
     * @param numOfFeatures
     *            this could be -1, if so, the default value will be
     *            len(features)^0.5
     * @param maxDepth
     *            this could be -1, if so, any leaf node will have only 1
     *            instance.
     */
    public void train(double[][] instances, int[] targets, int numOfTrees, int numOfFeatures,
            int maxDepth) {
        Util.CHECK(instances.length == targets.length, "");
        Util.CHECK(numOfTrees > 0, "");
        this.instances_ = instances;
        this.targets_ = targets;
        this.numOfTrees_ = numOfTrees;
        this.numOfFeatures_ = numOfFeatures;
        this.maxDepth_ = maxDepth;
        this.trees_ = new TreeNode[numOfTrees_];
        for (int i = 0; i < trees_.length; i++) {
            System.out.println("building the tree:" + i);
            trees_[i] = buildTree(getRandomInstances(400), 1);
        }
    }

    // Get sub set of all instances randomly.
    List<Integer> getRandomInstances(int numOfInstances) {
        List<Integer> ret = new ArrayList<Integer>(numOfInstances);
        Random rand = new Random();
        while (ret.size() < numOfInstances) {
            ret.add(rand.nextInt(instances_.length));
        }
        return ret;
    }

    // Get the majority class of all samples having indices.
    private int getMajorClass(List<Integer> indices) {
        Map<Integer, Integer> mii = new HashMap<Integer, Integer>();
        int best = -1;
        int ret = -1;
        for (int index : indices) {
            Integer v = mii.get(targets_[index]);
            if (v == null) {
                v = 0;
            }
            mii.put(targets_[index], v + 1);
            if (v + 1 > best) {
                best = v + 1;
                ret = targets_[index];
            }
        }
        return ret;
    }

    // Are all samples having indices have the same class?
    private boolean haveSameClass(List<Integer> indices) {
        for (int i = 1; i < indices.size(); i++) {
            if (targets_[indices.get(i)] != targets_[indices.get(0)]) {
                return false;
            }
        }
        return true;
    }

    // Get a list of indices of features randomly.
    private List<Integer> getRandomFeatures() {
        Random rand = new Random();
        Set<Integer> set = new HashSet<Integer>();
        int featureSize = instances_[0].length;
        while (set.size() < numOfFeatures_) {
            set.add(rand.nextInt(featureSize));
        }

        List<Integer> ret = new ArrayList<Integer>();
        ret.addAll(set);
        return ret;
    }

    // Get the entropy of some samples.
    private double getEntropy(List<Integer> indices, int from, int to) {
        Map<Integer, Integer> mii = new HashMap<Integer, Integer>();
        for (int i = from; i < to; i++) {
            Integer v = mii.get(indices.get(i));
            if (v == null) {
                v = 0;
            }
            mii.put(indices.get(i), v + 1);
        }
        double ret = 0;
        for (Integer key : mii.keySet()) {
            int v = mii.get(key);
            ret += Math.log((to - from) * 1.0 / v);
        }
        return ret;
    }

    private TreeNode buildTree(List<Integer> indices, int curDepth) {
        System.out.println("building tree, depth:" + curDepth);
        if (maxDepth_ == curDepth) {
            return new TreeNode(-1, -1, getMajorClass(indices), null, null, true);
        }
        if (haveSameClass(indices)) {
            return new TreeNode(-1, -1, targets_[indices.get(0)], null, null, true);
        }
        List<Integer> featureInices = getRandomFeatures();
        double bestEntropy = Double.MAX_VALUE;
        int bestFeatureIndex = -1;
        double splitValue = -1;
        List<Integer> leftIndices = null;
        List<Integer> rightIndices = null;

        for (final int featureIndex : featureInices) {
            Collections.sort(indices, new Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    if (instances_[o1][featureIndex] < instances_[o2][featureIndex]) {
                        return -1;
                    } else if (instances_[o1][featureIndex] == instances_[o2][featureIndex]) {
                        return 0;
                    } else {
                        return 1;
                    }
                }
            });
            for (int i = 0; i < indices.size() - 1; i++) {
                double entropy = 1.0 * (i + 1 - 0) / indices.size() * getEntropy(indices, 0, i + 1)
                        + 1.0 * (indices.size() - (i + 1)) / indices.size()
                        * getEntropy(indices, i + 1, indices.size());
                if (entropy < bestEntropy) {
                    bestEntropy = entropy;
                    bestFeatureIndex = featureIndex;
                    leftIndices = new ArrayList<Integer>();
                    rightIndices = new ArrayList<Integer>();
                    leftIndices.addAll(indices.subList(0, i + 1));
                    rightIndices.addAll(indices.subList(i + 1, indices.size()));
                    splitValue = instances_[indices.get(i)][featureIndex];
                }
            }
        }
        return new TreeNode(bestFeatureIndex, splitValue, 1, buildTree(leftIndices, curDepth + 1),
                buildTree(rightIndices, curDepth + 1), false);
    }

    private int predicateByOneTree(TreeNode node, double[] instance) {
        if (node.isLeafNode_) {
            return node.target_;
        }
        if (instance[node.featureIndex_] <= node.value_) {
            return predicateByOneTree(node.left_, instance);
        } else {
            return predicateByOneTree(node.right_, instance);
        }
    }

    // Predicate one instance.
    public int predicate(double[] instance) {
        Map<Integer, Integer> mii = new HashMap<Integer, Integer>();
        int bestTarget = -1;
        int bestCount = -1;
        for (TreeNode root : trees_) {
            int target = predicateByOneTree(root, instance);
            Integer v = mii.get(target);
            if (v == null) {
                v = 0;
            }
            mii.put(target, v + 1);
            if (v + 1 > bestCount) {
                bestCount = v + 1;
                bestTarget = target;
            }
        }
        return bestTarget;
    }

    // TreeNode of the decision tree.
    private static class TreeNode {
        public int featureIndex_;
        public double value_;
        public int target_;
        public TreeNode left_;
        public TreeNode right_;
        public boolean isLeafNode_;

        public TreeNode(int featureIndex, double value_, int target_, TreeNode left_,
                TreeNode right_, boolean isLeafNode) {
            this.featureIndex_ = featureIndex;
            this.value_ = value_;
            this.target_ = target_;
            this.left_ = left_;
            this.right_ = right_;
            this.isLeafNode_ = isLeafNode;
        }
    }

    public static void main(String[] args) {

    }
}
