import java.io.File
import java.io.PrintWriter
import java.util.StringTokenizer

private val INPUT = File("input.txt").inputStream()
private val OUTPUT = File("output.txt").outputStream()


private val bufferedReader = INPUT.bufferedReader()
private val outputWriter = PrintWriter(OUTPUT, false)
private fun readLn() = bufferedReader.readLine()!!

private fun readList() = readLn().split(' ')
private var tokenizer = StringTokenizer("")
private fun read(): String {
    while (tokenizer.hasMoreTokens().not()) tokenizer = StringTokenizer(readLn(), " ")
    return tokenizer.nextToken()
}

private fun readInt() = read().toInt()
private fun readLong() = read().toLong()
private fun readDouble() = read().toDouble()

private fun readIntList() = readList().map { it.toInt() }
private fun readLongList() = readList().map { it.toLong() }
private fun readDoubleList() = readList().map { it.toDouble() }

private fun readIntArray(n: Int = 0) =
    if (n == 0) readList().run { IntArray(size) { get(it).toInt() } } else IntArray(n) { readInt() }

private fun readLongArray(n: Int = 0) =
    if (n == 0) readList().run { LongArray(size) { get(it).toLong() } } else LongArray(n) { readLong() }

private fun readDoubleArray(n: Int = 0) =
    if (n == 0) readList().run { DoubleArray(size) { get(it).toDouble() } } else DoubleArray(n) { readDouble() }


private fun Int.modPositive(other: Int): Int = if (this % other < 0) ((this % other) + other) else (this % other)


private class task4 {
    fun solveTestCase(): String {
        //TODO: Solve the question
        val n = readLong()
        val s = readLongArray()

        if (s.all { it == 1L }) {
            return calculate(s.size.toLong()).toString()
        }
        var temp: Long = 0
        var result: Long = 0
        s.forEach {
            if (it == 1L) {
                result += calculate(temp)
                temp = 0
            } else
                temp++
        }
        result += calculate((temp))
        return result.toString()
    }

    private fun calculate(toLong: Long): Long {
        return toLong * (toLong + 1) / 2
    }

}


fun main(args: Array<String>) {


    outputWriter.println(
        task4()
            .solveTestCase()
    )


    outputWriter.flush()


}