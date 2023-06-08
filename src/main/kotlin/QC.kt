import java.io.*
import java.util.*


private val br = BufferedReader(InputStreamReader(System.`in`))
val writer = BufferedWriter(OutputStreamWriter(System.out))
private fun readInt() = br.readLine().toInt()
private fun readString() = br.readLine()
private fun readLong() = br.readLine().toLong()
private fun readDouble() = br.readLine().toDouble()


private fun readListString() = readString().split(' ')
private fun readListInt() = readString().split(' ').map { it.toInt() }
private fun readListLong() = readString().split(' ').map { it.toLong() }

private fun writeToScreen(str: String) = writer.write(str)

fun main() {

    val takingOff = readString()
    val landing = readString()
    val gmt = readString()
    val start = getTimeInMinutes(takingOff)
    val end = getTimeInMinutes(landing)
    val withGMTEndMinutes = calculateWithGMT(gmt, end)
    val abs = Math.abs(start - withGMTEndMinutes)
    println("${formatNum(abs / 60)}:${formatNum(abs % 60)}")

}

fun formatNum(num: Int): String {
    if ("$num".length == 2)
        return "$num"
    return "0$num"
}

fun calculateWithGMT(gmt: String, end: Int): Int {
    if (gmt == "0")
        return end
    val minutes = gmt.substring(1).toInt() * 60
    if (gmt[0] == '+') {
        if (minutes < end)
            return end - minutes
        return (24 * 60) + (end - minutes)
    }
    return (end + minutes) % (24 * 60)
}

fun getTimeInMinutes(str: String): Int {
    val (hour, minute) = str.split(":").map { it.toInt() }
    return hour * 60 + minute
}






















