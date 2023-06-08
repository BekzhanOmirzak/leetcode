import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter


private val br = BufferedReader(InputStreamReader(System.`in`))
private fun readInt() = br.readLine().toInt()
private fun readString() = br.readLine()
private fun readLong() = br.readLine().toLong()
private fun readDouble() = br.readLine().toDouble()


private fun readListString() = readString().split(' ')
private fun readListInt() = readString().split(' ').map { it.toInt() }
private fun readListLong() = readString().split(' ').map { it.toLong() }


fun main(args: Array<String>) {

    val reader = BufferedReader(InputStreamReader(System.`in`))
    val writer = BufferedWriter(OutputStreamWriter(System.out))

    val (n, k, w) = readListInt()
    val totalWeeks = w * n
    val list = mutableListOf<IntArray>()
    repeat(k) {
        val (cost, week) = readListInt()
        list.add(intArrayOf(cost, week))
    }
    val memo= hashMapOf<String,Long>()
    println(dp(totalWeeks,0,list.toTypedArray(),memo))

    reader.close()
    writer.close()
}

fun dp(total: Int, at: Int, ads: Array<IntArray>,memo:HashMap<String,Long>): Long {

    if (at == ads.size)
        return 0
    val key="$total|$at"
    if(memo.containsKey(key))
        return memo[key]!!
    var max = 0L
    val (c, w) = ads[at]
    if (w <= total) {
        val res=w*c+dp(total-w,at+1,ads,memo)
        max=Math.max(max,res)
    }
    max=Math.max(max,dp(total,at+1,ads,memo))
    memo[key]=max

    return max
}




























