import java.io.BufferedReader
import java.io.InputStreamReader


private val br = BufferedReader(InputStreamReader(System.`in`))
private fun readInt() = br.readLine().toInt()
private fun readString() = br.readLine()
private fun readLong() = br.readLine().toLong()
private fun readDouble() = br.readLine().toDouble()


private fun readListString() = readString().split(' ')
private fun readListInt() = readString().split(' ').map { it.toInt() }
private fun readListLong() = readString().split(' ').map { it.toLong() }


fun main() {



}

fun shift(str: String, shift: Long): String {

    val len = countLen(str)
    val mod = (shift % len).toInt()

    return findString(str, mod)
}

fun countLen(str: String): Long {
    var c = 0L
    var at = 0
    while (at < str.length) {
        var n = 1
        if (str[at].isDigit()) {
            n = str[at] - '0'
            at++
            if (str[at].isDigit()) {
                n = n * 10 + (str[at] - '0')
                at++
            }
        }
        c += n
        at++
    }
    return c
}

fun findString(str: String, shift: Int): String {

    val end = StringBuilder()
    var cur = shift
    val ans = StringBuilder()
    var at = 0
    while (at < str.length && cur != 0) {
        var n = 1
        if (str[at].isDigit()) {
            n = str[at] - '0'
            at++
            if (str[at].isDigit()) {
                n = n * 10 + (str[at] - '0')
                at++
            }
        }
        val char = str[at++]
        if (n <= cur) {
            end.append("$n$char")
            cur -= n
        } else {
            end.append("$cur$char")
            ans.append("${n - cur}$char")
            cur = 0
        }
    }
//    println("Ans Start : $ans")
    if (at == str.length)
        return str
    while (at < str.length) {
        ans.append(str[at])
        at++
    }
    at = 0
    var endNum = 1
    if (end[at].isDigit()) {
        endNum = end[at] - '0'
        at++
        if (end[at].isDigit()) {
            endNum = endNum * 10 + (end[at] - '0')
            at++
        }
    }
    val endChar = end[at]
    if (ans[ans.length - 1] != endChar)
        return ans.append(end).toString()
    var e = ans.length - 2
    var ansNum = 1
    if (ans[e].isDigit()) {
        ansNum = ans[e] - '0'
        if (ans[e].isDigit()) {
            e--
            ansNum = (ans[e] - '0') * 10 + ansNum
        }
    } else
        e++
//    println("After Triming Start : " + ans.substring(0, e))
    val trimed = end.substring(at + 1)
//    println("Trimed End : $trimed")

    return ans.substring(0, e) + "${ansNum + endNum}$endChar" + trimed
}























