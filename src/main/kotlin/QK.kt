import java.io.BufferedReader
import java.io.InputStreamReader

/*
10 3
6 7 g
7 3 y
3 10 r
1
6 10
 */

private val br = BufferedReader(InputStreamReader(System.`in`))
private fun readInt() = br.readLine().toInt()
private fun readString() = br.readLine()
private fun readLong() = br.readLine().toLong()
private fun readDouble() = br.readLine().toDouble()


private fun readListString() = readString().split(' ')
private fun readListInt() = readString().split(' ').map { it.toInt() }
private fun readListLong() = readString().split(' ').map { it.toLong() }

fun main() {

    val (cities, edges) = readListInt()
    val adj = Array(cities + 1) { mutableListOf<DestTo>() }
    repeat(edges) {
        val (from, to, color) = readString().split(" ")
        adj[from.toInt()].add(DestTo(to.toInt(), color[0]))
    }
    val colors = listOf('g', 'y', 'r')
    val questions = readInt()
    val cached = hashMapOf<String, Int>()
    val memo = hashMapOf<String, Int>()
    repeat(questions) {
        val (from, to) = readListInt()
        val key = "$from|$to"
        if (cached.containsKey(key)) {
            println(cached[key])
        } else {
            val res = dfs(from, 0, adj, to, colors, memo)
            cached[key] = res
            println(res)
        }
    }

}

fun dfs(
    at: Int,
    atColor: Int,
    adj: Array<MutableList<DestTo>>,
    to: Int,
    colors: List<Char>,
    memo: HashMap<String, Int>
): Int {
    if (atColor == colors.size && at == to)
        return 1
    if (atColor == colors.size)
        return 0
    val key = "$at|$atColor|$to"
    if (memo.containsKey(key))
        return memo[key]!!
    var cur = 0
    for (n in adj[at]) {
        if (n.color == colors[atColor]) {
            cur += dfs(n.to, atColor + 1, adj, to, colors, memo)
        }
    }
    memo[key] = cur
    return cur
}

data class DestTo(val to: Int, val color: Char)





















