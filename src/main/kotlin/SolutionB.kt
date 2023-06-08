import java.io.BufferedReader
import java.io.InputStreamReader

private val br = BufferedReader(InputStreamReader(System.`in`))
private fun readInt() = br.readLine().toInt()
private fun readStringList() = br.readLine().split(" ")
private fun readIntList() = readStringList().map { it.toInt() }

fun main() {
    val t = readInt()
    repeat(t) {
        val (n, k) = readIntList()
        val a = readIntList()
        val b = readIntList()

        val sortedB = b.sorted().toMutableList()
        val indexMap = sortedB.withIndex().associate { it.value to it.index }

        val result = mutableListOf<Int>()
        for ((i, ai) in a.withIndex()) {
            val closestElement = sortedB.minByOrNull { Math.abs(ai - it) }!!
            val index = indexMap[closestElement]!!
            result.add(sortedB[index])
            sortedB.removeAt(index)
        }

        println(result.joinToString(" "))
    }
}
