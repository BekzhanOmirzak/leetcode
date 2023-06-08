import java.util.Scanner;

public class SolutionA {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int utc = Integer.parseInt(scanner.nextLine());
        System.out.println(solve(utc));
        scanner.close();
    }

    public static String solve(int utc) {
        if (utc == 6) {
            return "11:00";
        } else if (utc > 6) {
            int hour = 11 + (utc - 6);
            return hour + ":00";
        }
        int hour = 11 + (utc - 6);
        if (hour >= 0) {
            String str = String.valueOf(hour);
            if (str.length() == 1) {
                str = "0" + str;
            }
            return str + ":00";
        }
        return (24 + hour) + ":00";
    }
}
