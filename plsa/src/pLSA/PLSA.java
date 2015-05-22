package pLSA;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Scanner;

public class PLSA {
    static Random rand = new Random();

    public static class WordBelief {
        public int i;
        public double d;

        public String toString() {
            return "" + i + ":" + d;
        }
    }

    public static ArrayList<WordBelief> make(double[] array) {
        ArrayList<WordBelief> ret = new ArrayList<WordBelief>();
        for (int i = 0; i < array.length; i++) {
            WordBelief idp = new WordBelief();
            idp.d = array[i];
            idp.i = i;
            ret.add(idp);
        }
        return ret;
    }

    public static void sort(ArrayList<WordBelief> list) {
        Collections.sort(list, new Comparator<WordBelief>() {
            @Override
            public int compare(WordBelief o1, WordBelief o2) {
                if (o1.d == o2.d) {
                    return 0;
                } else if (o1.d < o2.d) {
                    return 1;
                } else {
                    return -1;
                }
            }
        });
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

    static class DoubleArray {
        public double[] v;

        public DoubleArray(double[] v) {
            this.v = v;
        }
    }

    static class DocWord {
        public int doc;
        public int word;

        public DocWord(int doc, int word) {
            this.doc = doc;
            this.word = word;
        }

        @Override
        public int hashCode() {
            return doc + word;
        }

        @Override
        public boolean equals(Object obj) {
            DocWord dw = (DocWord) obj;
            return doc == dw.doc && word == dw.word;
        }

        public String toString() {
            return "" + doc + "," + word;
        }
    }

    static HashMap<Integer, String> wordIdToWord = new HashMap<Integer, String>();
    static HashMap<String, Integer> wordTowordId = new HashMap<String, Integer>();
    static HashMap<Integer, HashMap<Integer, Integer>> docWordFreq = new HashMap<Integer, HashMap<Integer, Integer>>();
    static HashMap<Integer, HashMap<Integer, Integer>> wordDocFreq = new HashMap<Integer, HashMap<Integer, Integer>>();
    static HashMap<DocWord, DoubleArray> docWordTopicP = new HashMap<DocWord, DoubleArray>();

    static double[][] theta;
    static double[][] phi;
    static int words = 0;
    static int docs = 0;
    static int topics = 2;
    static int iterations = 30;

    public static void main(String[] args) throws FileNotFoundException {
        Scanner scanner = new Scanner(new File("data/test.txt"));
        // The first line is useless.
        scanner.nextLine();
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            docWordFreq.put(docs, new HashMap<Integer, Integer>());
            String[] tokens = line.split(" ");
            for (String token : tokens) {
                if (!wordTowordId.containsKey(token)) {
                    wordTowordId.put(token, words);
                    wordIdToWord.put(words, token);
                    words++;
                }
                int wordId = wordTowordId.get(token);
                if (!wordDocFreq.containsKey(wordId)) {
                    wordDocFreq.put(wordId, new HashMap<Integer, Integer>());
                }
                Integer freq1 = wordDocFreq.get(wordId).get(docs);
                if (freq1 == null) {
                    wordDocFreq.get(wordId).put(docs, 1);
                } else {
                    wordDocFreq.get(wordId).put(docs, freq1 + 1);
                }

                Integer freq2 = docWordFreq.get(docs).get(wordId);
                if (freq2 == null) {
                    docWordFreq.get(docs).put(wordId, 1);
                } else {
                    docWordFreq.get(docs).put(wordId, freq2 + 1);
                }
            }
            docs++;
        }
        scanner.close();

        theta = new double[docs][topics];
        phi = new double[topics][words];

        for (int i = 0; i < docs; i++) {
            for (int k = 0; k < topics; k++) {
                theta[i][k] = rand.nextDouble() + 0.5;
            }
            normalize(theta[i]);
        }

        for (int k = 0; k < topics; k++) {
            for (int j = 0; j < words; j++) {
                phi[k][j] = rand.nextDouble() + 0.5;
            }
            normalize(phi[k]);
        }

        for (int doc : docWordFreq.keySet()) {
            for (int word : docWordFreq.get(doc).keySet()) {
                docWordTopicP.put(new DocWord(doc, word), new DoubleArray(new double[topics]));
            }
        }

        for (int it = 0; it < iterations; it++) {
            System.out.println("iteration:" + it);
            // E-step
            for (int doc : docWordFreq.keySet()) {
                for (int word : docWordFreq.get(doc).keySet()) {
                    DoubleArray topicsP = docWordTopicP.get(new DocWord(doc, word));
                    for (int k = 0; k < topics; k++) {
                        topicsP.v[k] = theta[doc][k] * phi[k][word];
                    }
                    normalize(topicsP.v);
                }
            }

            // M-step
            for (int doc = 0; doc < docs; doc++) {
                HashMap<Integer, Integer> wordFreqMap = docWordFreq.get(doc);
                for (int topic = 0; topic < topics; topic++) {
                    double d = 0;
                    for (Entry<Integer, Integer> wordFreq : wordFreqMap.entrySet()) {
                        d += wordFreq.getValue()
                                * docWordTopicP.get(new DocWord(doc, wordFreq.getKey())).v[topic];
                    }
                    theta[doc][topic] = d;
                }
                normalize(theta[doc]);
            }
            for (int topic = 0; topic < topics; topic++) {
                for (int word = 0; word < words; word++) {
                    double d = 0;
                    for (Entry<Integer, Integer> docFreq : wordDocFreq.get(word).entrySet()) {
                        d += docFreq.getValue()
                                * docWordTopicP.get(new DocWord(docFreq.getKey(), word)).v[topic];
                    }
                    phi[topic][word] = d;
                }
                normalize(phi[topic]);
            }
        }

        for (int topic = 0; topic < topics; topic++) {
            ArrayList<WordBelief> topicWords = make(phi[topic]);
            sort(topicWords);

            for (int word = 0; word < Math.min(20, topicWords.size()); word++) {
                if (topicWords.get(word).d > 0.00000000001) {
                    System.out.print(wordIdToWord.get(topicWords.get(word).i) + " "
                            + String.format("%.3f", topicWords.get(word).d) + ",");
                }
            }
            System.out.println();
        }
    }
}
