package faen;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class CsvToArff {

    private static String cv(String s) {
        if (s.isEmpty()) {
            return "?";
        }
        return s;
    }

    public static void convertTrain() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(
                "/Users/zhangfaen/dev/ml/kaggle/titanic/data/train.csv"));
        BufferedWriter bw = new BufferedWriter(new FileWriter(
                "/Users/zhangfaen/dev/ml/kaggle/titanic/data/train.arff"));
        bw.write("@relation titantic\n");
        // index 2
        bw.write("@attribute pclass {1,2,3} \n");
        // index 4
        bw.write("@attribute sex {male,female} \n");
        // index 5
        bw.write("@attribute age numeric \n");
        // index 6
        bw.write("@attribute SibSp numeric \n");
        // index 7
        bw.write("@attribute Parch numeric \n");
        // index 8
        // bw.write("@attribute Ticket numeric \n");
        // index 9
        bw.write("@attribute Fare numeric \n");
        // index 10
        bw.write("@attribute Cabin numeric \n");
        // index 11
        bw.write("@attribute Embarked {Q,C,S} \n");
        // index 1
        bw.write("@attribute class {0,1} \n");
        bw.write("@data\n");
        int total = 0;
        while (true) {
            String s = br.readLine();
            if (s == null) {
                break;
            }
            if (total++ == 0) {
                // PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
                continue;
            }
            String t = "";
            boolean opened = false;
            for (char c : s.toCharArray()) {
                if (c == '"') {
                    opened = !opened;
                }
                if (c == ',' && opened) {

                } else {
                    t += c;
                }
            }
            s = t;
            String[] sa = s.split(",", -1);
            if (sa.length != 12) {
                System.out.println(s);
            }
            bw.write(cv(sa[2]) + "," + cv(sa[4]) + "," + cv(sa[5]) + "," + cv(sa[6]) + ","
                    + cv(sa[7]) + "," + cv(sa[9]) + "," + sa[10].split(" ").length + ","
                    + cv(sa[11]) + "," + cv(sa[1]) + "\n");
        }
        br.close();
        bw.close();
    }

    public static void convertTest() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(
                "/Users/zhangfaen/dev/ml/kaggle/titanic/data/test.csv"));
        BufferedWriter bw = new BufferedWriter(new FileWriter(
                "/Users/zhangfaen/dev/ml/kaggle/titanic/data/test.arff"));
        bw.write("@relation titantic\n");
        // index 2
        bw.write("@attribute pclass {1,2,3} \n");
        // index 4
        bw.write("@attribute sex {male,female} \n");
        // index 5
        bw.write("@attribute age numeric \n");
        // index 6
        bw.write("@attribute SibSp numeric \n");
        // index 7
        bw.write("@attribute Parch numeric \n");
        // index 8
        // bw.write("@attribute Ticket numeric \n");
        // index 9
        bw.write("@attribute Fare numeric \n");
        // index 10
        bw.write("@attribute Cabin numeric \n");
        // index 11
        bw.write("@attribute Embarked {Q,C,S} \n");
        // index 1
        bw.write("@attribute class {0,1} \n");
        bw.write("@data\n");
        int total = 0;
        while (true) {
            String s = br.readLine();
            if (s == null) {
                break;
            }
            if (total++ == 0) {
                // PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
                continue;
            }
            int firstComma = s.indexOf(",");
            s = s.substring(0, firstComma) + ",1" + s.substring(firstComma, s.length());
            String t = "";
            boolean opened = false;
            for (char c : s.toCharArray()) {
                if (c == '"') {
                    opened = !opened;
                }
                if (c == ',' && opened) {

                } else {
                    t += c;
                }
            }
            s = t;
            String[] sa = s.split(",", -1);
            if (sa.length != 12) {
                System.out.println(s);
            }
            bw.write(cv(sa[2]) + "," + cv(sa[4]) + "," + cv(sa[5]) + "," + cv(sa[6]) + ","
                    + cv(sa[7]) + "," + cv(sa[9]) + "," + sa[10].split(" ").length + ","
                    + cv(sa[11]) + "," + cv(sa[1]) + "\n");
        }
        br.close();
        bw.close();
    }

    public static void main(String[] args) throws IOException {
        convertTrain();
        convertTest();
    }

}
