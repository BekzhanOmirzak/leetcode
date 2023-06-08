import java.util.Scanner;

public class SolutionD {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        String[] input1 = scanner.nextLine().split(" ");
        long city = Long.parseLong(input1[0]);
        long people = Long.parseLong(input1[1]);

        String[] input2 = scanner.nextLine().split(" ");
        int[] capacities = new int[input2.length];
        for (int i = 0; i < input2.length; i++) {
            capacities[i] = Integer.parseInt(input2[i]);
        }

        String[] input3 = scanner.nextLine().split(" ");
        int[] times = new int[input3.length];
        for (int i = 0; i < input3.length; i++) {
            times[i] = Integer.parseInt(input3[i]);
        }

        long total = 0L;
        for (int t : times) {
            total += t;
        }

        int minCap = Integer.MAX_VALUE;
        for (int capacity : capacities) {
            minCap = Math.min(minCap, capacity);
        }

        long counter = people/minCap;
        if(people%minCap==0L)
            counter--;

        System.out.println(counter + total);
    }
}
