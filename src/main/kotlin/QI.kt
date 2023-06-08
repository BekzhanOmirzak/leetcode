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

    readInt()
    val nums = readListInt().toIntArray()
    val odds = mutableListOf<Int>()
    val evens = mutableListOf<Int>()
    for (n in nums) {
        if (n % 2 == 0)
            evens.add(n)
        else
            odds.add(n)
    }
    val abs = Math.abs(odds.size - evens.size)
    if (abs >= 2) {
        println(-1)
        return
    }
    val clone = nums.clone()

    var  odd = Int.MAX_VALUE
    if (odds.size >= evens.size) {
        odd = startOdd(clone)
    }
    var even = Int.MAX_VALUE
    if (evens.size >= odds.size) {
        even = startEven(nums)
    }
    if (odd < even) {
        println(odd)
        println(clone.joinToString(" "))
    } else {
        println(even)
        println(nums.joinToString(" "))
    }

}

fun startOdd(nums: IntArray): Int {
    var c = 0
    val odds = mutableListOf<Int>()
    val evens = mutableListOf<Int>()
    for (i in 0 until nums.size step 2) {
        if (nums[i] % 2 != 1) {
            c++
            evens.add(nums[i])
        }
        if (i + 1 < nums.size && nums[i + 1] % 2 != 0) {
            c++
            odds.add(nums[i + 1])
        }
    }
    var atOdd = 0
    var atEven = 0
    for (i in 0 until nums.size step 2) {
        if (nums[i] % 2 != 1) {
            nums[i] = odds[atOdd]
            atOdd++
        }
        if (i + 1 < nums.size && nums[i + 1] % 2 != 0) {
            nums[i + 1] = evens[atEven]
            atEven++
        }
    }
    return c
}

fun startEven(nums: IntArray): Int {
    var c = 0
    val odds = mutableListOf<Int>()
    val evens = mutableListOf<Int>()
    for (i in 0 until nums.size step 2) {
        if (nums[i] % 2 != 0) {
            c++
            odds.add(nums[i])
        }
        if (i + 1 < nums.size && nums[i + 1] % 2 != 1) {
            c++
            evens.add(nums[i + 1])
        }
    }
    var atOdd = 0
    var atEven = 0
    for (i in 0 until nums.size step 2) {
        if (nums[i] % 2 != 0) {
            nums[i] = evens[atEven]
            atEven++
        }
        if (i + 1 < nums.size && nums[i + 1] % 2 != 0) {
            nums[i + 1] = odds[atOdd]
            atOdd++
        }
    }
    return c
}
























