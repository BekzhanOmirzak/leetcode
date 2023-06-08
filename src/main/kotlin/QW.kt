import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.io.PrintWriter
import java.util.StringTokenizer


private val br = BufferedReader(InputStreamReader(System.`in`))
private fun readInt() = br.readLine().toInt()
private fun readString() = br.readLine()
private fun readLong() = br.readLine().toLong()
private fun readDouble() = br.readLine().toDouble()


private fun readListString() = readString().split(' ')
private fun readListInt() = readString().split(' ').map { it.toInt() }
private fun readListLong() = readString().split(' ').map { it.toLong() }


fun main() {

    val str = readString()
    val used = mutableSetOf<String>()
    dfs(0, str, used, "", 0)
    println(ansStr)

}

var ansStr = ""
var max = 0

fun dfs(at: Int, str: String, used: MutableSet<String>, ans: String, c: Int) {

    if (at >= str.length) {
        if (c > max && ans.isNotEmpty()) {
            max = c
            ansStr = ans.substring(1)
        }
        return
    }
    val cur = StringBuilder()
    var i = at
    while (cur.isEmpty() || (i < str.length && used.contains(cur.toString()))) {
        cur.append(str[i])
        i++
        if (!used.contains(cur.toString()) && !(cur.length >= 2 && cur[0] == '0')) {
            used.add(cur.toString())
            println("Cur : $cur")
            dfs(i, str, used, "$ans-$cur", c + 1)
            used.remove(cur.toString())
        }
    }

}

























