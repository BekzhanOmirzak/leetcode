import java.math.BigInteger
import java.util.*
import kotlin.Comparator
import kotlin.collections.HashMap
import kotlin.random.Random


/*


1 + 1 = 0, carry -1
-1 + 0 = 1, carry 1
1 + 0 = 1, carry 0
0 + 0 = 0, carry 0
0 + 1 = 1, carry 0
-1 + 1 = 0, carry 0
[
"..#....##.",
"....d.#.D#",
"#...#.c...",
"..##.#..a.",
"...#....##",
"#....b....",
".#..#.....",
"..........",
".#..##..A.",
".B..C.#..@"
]

[["E","E","E","E","E","E","E","E"],
["E","E","E","E","E","E","E","M"],
["E","E","M","E","E","E","E","E"],
["M","E","E","E","E","E","E","E"],
["E","E","E","E","E","E","E","E"],
["E","E","E","E","E","E","E","E"],
["E","E","E","E","E","E","E","E"],
["E","E","M","M","E","E","E","E"]]
[0,0]

[["B","B","B","B","B","B","1","E"],
["B","1","1","1","B","B","1","M"],
["1","2","M","1","B","B","1","1"],
["M","2","1","1","B","B","B","B"],
["1","1","B","B","B","B","B","B"],
["B","B","B","B","B","B","B","B"],
["B","1","2","2","1","B","B","B"],
["B","1","M","M","1","B","B","B"]]

[
["B","B","B","B","B","B","1","E"],
["B","1","1","1","B","B","1","M"],
["1","E","M","1","B","B","1","1"],
["M","E","1","1","B","B","B","B"],
["1","1","B","B","B","B","B","B"],
["B","B","B","B","B","B","B","B"],
["B","1","2","2","1","B","B","B"],
["B","1","M","M","1","B","B","B"]
]
sout

[["B","1","E","1","B"],
["B","1","M","1","B"],
["B","1","1","1","B"],
["B","B","B","B","B"]]
[1,1]
[5,10]
[[3,4,5,2,5],[4,5,3,8,3],[3,2,5,3,1]]
[1,2,2,1,1]
[[0,2],[0,4],[1,4],[2,3]]

    var a=3
    var b=5
    b=b-a //b=2
    a=a+b //a=5
    b=a-b //b=3
    println("a : $a b : $b")
 */

val mod = Math.pow(10.0, 9.0).toInt() + 7
val dirs = arrayOf(
    intArrayOf(1, 0), intArrayOf(0, 1), intArrayOf(-1, 0), intArrayOf(0, -1)
)

fun findGCD(i: Int, j: Int): Int {
    var bigger = i
    var small = j
    while (small != 0) {
        val t = bigger
        bigger = small
        small = t % small
    }
    return bigger
}

val cachPrime = hashMapOf<Int, Boolean>()

fun isPrime(num: Int): Boolean {
    if (cachPrime.containsKey(num)) return cachPrime[num]!!
    if (num == 2) return true
    else if (num == 1) return false
    val sqrt = Math.sqrt(num.toDouble()).toInt()
    for (i in 2..sqrt) {
        if (num % i == 0) {
            cachPrime[num] = false
            return false
        }
    }
    cachPrime[num] = true
    return true
}


fun main() {


    println(
        matrixSumQueries(
            3,
            arrayOf(
                intArrayOf(0, 0, 1),
                intArrayOf(1, 2, 2),
                intArrayOf(0, 2, 3),
                intArrayOf(1, 0, 4)
            )
        )
    )

    println(
        count(
            "1",
            "5",
            1,
            5
        )
    )



    println(
        shortestCommonSupersequence(
            "abac",
            "cab"
        )
    )

}

fun shortestCommonSupersequence(str1: String, str2: String): String {

    val memo = hashMapOf<String, String>()
    val res = dpExplore(0, 0, str1, str2, memo)
    val ans = StringBuilder()
    var at1 = 0
    var at2 = 0
    var at = 0
    while (at1 < str1.length && at2 < str2.length && at < res.length) {
        if (str1[at1] == str2[at2] && res[at] == str1[at1]) {
            ans.append(str1[at1])
            at1++
            at2++
            at++
            continue
        }
        if (str1[at1] != res[at]) {
            ans.append(str1[at1])
            at1++
        }
        if (str2[at2] != res[at]) {
            ans.append(str2[at2])
            at2++
        }
    }
    while (at1 < str1.length) {
        ans.append(str1[at1])
        at1++
    }
    while (at2 < str2.length) {
        ans.append(str2[at2])
        at2++
    }

    return ans.toString()
}

fun dpExplore(at1: Int, at2: Int, str1: String, str2: String, memo: HashMap<String, String>): String {
    if (at1 == str1.length || at2 == str2.length)
        return ""
    val key = "$at1|$at2"
    if (memo.containsKey(key))
        return memo[key]!!
    if (str1[at1] == str2[at2])
        return str1[at1] + dpExplore(at1 + 1, at2 + 1, str1, str2, memo)
    val left = dpExplore(at1 + 1, at2, str1, str2, memo)
    val right = dpExplore(at1, at2 + 1, str1, str2, memo)
    val res = if (left.length < right.length) right else left
    memo[key] = res
    return res
}


fun count(num1: String, num2: String, min_sum: Int, max_sum: Int): Int {


    return 0
}

fun matrixSumQueries(n: Int, queries: Array<IntArray>): Long {


    return 0L
}

fun semiOrderedPermutation(nums: IntArray): Int {
    var l = 0
    var r = 0
    for (i in 0 until nums.size) {
        if (nums[i] == 1) {
            l = i
        } else if (nums[i] == nums.size) {
            r = i
        }
    }
    var ans = l + (nums.size - r - 1)
    if (l > r)
        ans--
    return ans
}

fun findAnswer(nums: IntArray): Int {

    println(
        findAnswer(
            intArrayOf(
                -7, 3, 8, 12, -3, -10
            )
        )
    )

    return 0
}

fun maximumANDSum(nums: IntArray, numSlots: Int): Int {
    println(
        maximumANDSum(
            intArrayOf(2, 3), 3
        )
    )
    return 0
}

fun dpAndSum(atSlot: Int, numBit: Int, nums: IntArray, numSlot: Int, targetNumBit: Int): Int {
    if (atSlot > numSlot) return 0
    if (numBit == targetNumBit) return 0
    var max = 0



    return max
}

fun goodTriplets(nums1: IntArray, nums2: IntArray): Long {

    val size = nums1.size
    println(
        goodTriplets(
            intArrayOf(4, 0, 1, 3, 2), intArrayOf(4, 1, 0, 2, 3)
        )
    )


    return 0L
}


class SGTreeNum(val size: Int) {

    val board = IntArray(getSegmentSize())

    fun initTree(at: Int, l: Int, r: Int) {
        if (l == r) {
            board[at]++
            return
        }
        board[at] = r - l + 1
        val mid = (l + r) / 2
        initTree(at * 2 + 1, l, mid)
        initTree(at * 2 + 2, mid + 1, r)
    }

    private fun getSegmentSize(): Int {
        var cur = 1
        while (cur < size) {
            cur *= 2
        }
        return 2 * cur - 1
    }

    fun include(num: Int) {
        dfs(0, 0, size - 1, num, 1)
    }

    fun exclude(num: Int) {
        dfs(0, 0, size - 1, num, -1)
    }

    private fun dfs(at: Int, l: Int, r: Int, num: Int, inc: Int) {
        if (num < l || r < num) return
        if (l == r) {
            board[at] += inc
            return
        }
        board[at] += inc
        val mid = (l + r) / 2
        dfs(at * 2 + 1, l, mid, num, inc)
        dfs(at * 2 + 2, mid + 1, r, num, inc)
    }

    fun getRange(ql: Int, qr: Int): Int {
        return getRangeHelper(0, 0, size - 1, ql, qr)
    }

    private fun getRangeHelper(at: Int, l: Int, r: Int, ql: Int, qr: Int): Int {
        if (r < ql || qr < l) return 0
        if (ql <= l && r <= qr) {
            return board[at]
        }
        val mid = (l + r) / 2
        var c = 0
        c += getRangeHelper(at * 2 + 1, l, mid, ql, qr)
        c += getRangeHelper(at * 2 + 2, mid + 1, r, ql, qr)
        return c
    }

}

fun countPairs(nums: IntArray, k: Int): Long {

    var ans = 0L
    var size = nums.size - 1
    for (n in nums) {
        if (n % k == 0) {
            ans += size
            size--
        }
    }

    return ans
}

fun minimumFinishTime(tires: Array<IntArray>, changeTime: Int, numLaps: Int): Int {
    println(
        minimumFinishTime(
            arrayOf(
                intArrayOf(2, 3)
            ), 5, 4
        )
    )
    return dpExplore(0, numLaps, tires, changeTime)
}

fun dpExplore(at: Int, laps: Int, tires: Array<IntArray>, changeTime: Int): Int {

    if (at == tires.size || laps == 0) {
        if (laps == 0) return 0
        return Int.MAX_VALUE
    }
    var power = 1
    var min = dpExplore(at + 1, laps, tires, changeTime)
    var left = laps
    val tire = tires[at]
    var time = 0
    while (left > 0) {
        println("Time : ${tire[0] * power}")
        time += (tire[0] * power)
        left--
        power *= tire[1]
        val moveNext = dpExplore(at + 1, left, tires, changeTime)
        val stay = dpExplore(at, left, tires, changeTime)
        if (moveNext != Int.MAX_VALUE) {
            min = Math.min(min, moveNext + time + changeTime)
        }
        min = Math.min(min, stay + time + changeTime)
    }

    return min
}

fun maxStrength(nums: IntArray): Long {

    var res = 1L
    var largestNeg = Long.MIN_VALUE
    var largestPos = Long.MIN_VALUE
    var c = 0
    for (i in 0 until nums.size) {
        if (nums[i] != 0) {
            res *= nums[i]
            if (nums[i] < 0) {
                c++
                largestNeg = Math.max(largestNeg, nums[i].toLong())
                res *= nums[i].toLong()
            }
        }
        largestPos = Math.max(largestPos, nums[i].toLong())
    }
    if (c == 1 && largestPos <= 0) return largestPos
    if (c % 2 == 0) return res

    return res / largestNeg
}

fun lps(str: String): IntArray {

    println(
        lps(
            "aaaa"
        ).toList()
    )
    val lps = IntArray(str.length) { 0 }
    var l = 0
    var c = 0
    var r = 1
    while (r < str.length) {
        if (str[l] == str[r]) {
            lps[r] = ++c
            l++
            r++
        } else if (l > 0) {
            l--
            c = 0
        } else r++
    }

    return lps
}

fun maximumTop(nums: IntArray, k: Int): Int {

    if (nums.size == 1) {
        if (k % 2 == 0) return nums[0]
        return -1
    }
    if (k == 1) return nums[1]
    var max = nums[Math.min(nums.size - 1, k)]
    for (i in 0 until k) {
        if (i != k - 1) max = Math.max(max, nums[i])
    }

    return max
}

fun minimumWeight(n: Int, edges: Array<IntArray>, src1: Int, src2: Int, dest: Int): Long {

    println(
        minimumWeight(
            6, arrayOf(
                intArrayOf(0, 1, 1),
                intArrayOf(0, 2, 2),
                intArrayOf(1, 3, 3),
                intArrayOf(2, 1, 3),
                intArrayOf(2, 5, 7),
                intArrayOf(2, 3, 4),
                intArrayOf(3, 4, 5),
                intArrayOf(4, 5, 6),
            ), 0, 1, 5
        )
    )

    val adj = Array(n) { mutableListOf<IntArray>() }
    for ((from, to, cost) in edges) {
        adj[from].add(intArrayOf(to, cost))
    }
    val res = solve(src1, src2, dest, adj, n)
    if (res == -1L) return -1L

    return Math.min(res, solve(src2, src1, dest, adj, n))
}

fun solve(src1: Int, scr2: Int, dest: Int, adj: Array<MutableList<IntArray>>, n: Int): Long {

    val dist = LongArray(n) { Long.MAX_VALUE }
    val queue = PriorityQueue(object : java.util.Comparator<CurState> {
        override fun compare(o1: CurState, o2: CurState): Int {
            if (o1.cost < o2.cost) return -1
            return 0
        }
    })
    queue.add(CurState(src1, 0))
    dist[src1] = 0L
    val parents = IntArray(n) { -1 }
    while (queue.isNotEmpty()) {
        val poll = queue.poll()
        val neighs = adj[poll.at]
        for ((to, cost) in neighs) {
            val newCost = poll.cost + cost
            if (newCost < dist[to]) {
                dist[to] = newCost
                parents[to] = poll.at
                queue.add(CurState(to, newCost))
            }
        }
    }
    if (dist[dest] == Long.MAX_VALUE) return -1
    val path = mutableListOf<Int>()
    var cur = parents[dest]
    path.add(dest)
    while (cur != -1) {
        path.add(cur)
        cur = parents[cur]
    }
    path.reverse()
    val newDist = LongArray(n) { Long.MAX_VALUE }
    println(path)
    for (p in path) {
        newDist[p] = dist[p]
    }
    if (newDist[scr2] != Long.MAX_VALUE) return newDist[dest]
    queue.clear()
    queue.add(CurState(scr2, 0L))
    while (queue.isNotEmpty()) {
        val poll = queue.poll()
        val neighs = adj[poll.at]
        for ((to, cost) in neighs) {
            val newCost = poll.cost + cost
            if (newDist[to] != Long.MAX_VALUE) {
                return newCost + dist[dest]
            }
            if (newCost < newDist[to]) {
                newDist[to] = newCost
                queue.add(CurState(to, newCost))
            }
        }
    }

    return -1
}

data class CurState(val at: Int, val cost: Long)

fun minimumWhiteTiles(floor: String, numCarpets: Int, carpetLen: Int): Int {
    println(
        minimumWhiteTiles(
            "0110110011", 2, 2
        )
    )
    val dp = Array(numCarpets + 1) { IntArray(floor.length) { Int.MAX_VALUE } }
    var c = 0
    for (i in 0 until floor.length) {
        if (floor[i] == '1') c++
        dp[0][i] = c
    }
    for (car in 1..numCarpets) {
        for (i in 0 until floor.length) {
            if (i - carpetLen < 0) {
                dp[car][i] = 0
            } else {
                dp[car][i] = Math.min(dp[car][i], dp[car - 1][i - carpetLen])
            }
        }
    }
    for (d in dp) println(d.toList())

    return 0
}

fun dpExplore(at: Int, floor: String, carPet: Int, len: Int, memo: HashMap<String, Int>, suffix: IntArray): Int {

    if (at >= floor.length) return 0
    if (floor[at] == '0') return dpExplore(at + 1, floor, carPet, len, memo, suffix)
    if (carPet == 0) {
        return suffix[at]
    }
    val key = "$at|$carPet"
    if (memo.containsKey(key)) return memo[key]!!
    var min = dpExplore(at + len, floor, carPet - 1, len, memo, suffix)
    var c = 0
    for (i in at until floor.length) {
        if (floor[i] == '1') c++
        val res = c + dpExplore(i + 1 + len, floor, carPet - 1, len, memo, suffix)
        min = Math.min(min, res)
    }
    memo[key] = min

    return min
}

fun maxIncreasingCells(mat: Array<IntArray>): Int {

    val sortedRows = Array(mat.size) { listOf<Temp2>() }
    for (r in 0 until mat.size) {
        val curRows = mutableListOf<Temp2>()
        for (c in 0 until mat[r].size) {
            curRows.add(Temp2(mat[r][c], r, c))
        }
        curRows.sortWith(object : java.util.Comparator<Temp2> {
            override fun compare(o1: Temp2, o2: Temp2): Int {
                return o1.value - o2.value
            }
        })
        sortedRows[r] = curRows
    }
    val sortedCols = Array(mat[0].size) { listOf<Temp2>() }
    for (c in 0 until mat[0].size) {
        val cols = mutableListOf<Temp2>()
        for (r in 0 until mat.size) {
            cols.add(Temp2(mat[r][c], r, c))
        }
        cols.sortWith(object : Comparator<Temp2> {
            override fun compare(o1: Temp2, o2: Temp2): Int {
                return o1.value - o2.value
            }
        })
        sortedCols[c] = cols
    }
    val memo = Array(mat.size) { IntArray(mat[0].size) { -1 } }
    var ans = 1
    for (i in 0 until mat.size) {
        for (j in 0 until mat[i].size) {
            val cur = dpCur(i, j, mat, memo, sortedRows, sortedCols)
            ans = Math.max(ans, cur)
        }
    }

    return ans
}

data class Temp2(val value: Int, val i: Int, val j: Int)

fun dpCur(
    i: Int, j: Int, mat: Array<IntArray>, memo: Array<IntArray>, rows: Array<List<Temp2>>, cols: Array<List<Temp2>>
): Int {

    if (memo[i][j] != -1) return memo[i][j]
    var max = 1
    val cur = mat[i][j]

    val sRow = rows[i]
    val sCol = cols[j]
    var start = bs(cur + 1, sRow)
    for (k in start until sRow.size) {
        if (sRow[k].value > cur) {
            val ob = sRow[k]
            max = Math.max(max, 1 + dpCur(ob.i, ob.j, mat, memo, rows, cols))
        }
    }
    start = bs(cur + 1, sCol)
    for (k in start until sCol.size) {
        if (sCol[k].value > cur) {
            val ob = sCol[k]
            max = Math.max(max, 1 + dpCur(ob.i, ob.j, mat, memo, rows, cols))
        }
    }
    memo[i][j] = max

    return max
}

fun bs(min: Int, list: List<Temp2>): Int {
    var l = 0
    var r = list.size - 1
    var ans = list.size - 1
    while (l <= r) {
        val mid = (l + r) / 2
        if (list[mid].value >= min) {
            ans = Math.min(ans, mid)
            r = mid - 1
        } else l = mid + 1
    }
    return ans
}

fun minimumCost(s: String): Long {
    println(
        minimumCost(
            "010101"
        )
    )
    var zero = if (s[0] == '0') 0L else 1L
    var c = if (s[0] == '0') 0 else 1
    for (i in 1 until s.length - 1) {
        if (s[i] == '0') {
            if (c % 2 == 1) {
                c++
                zero += i
            }
        } else {
            if (c % 2 == 0) {
                c++
                zero += i
            }
        }
    }
    var ones = if (s[0] == '1') 0L else 1L
    c = if (s[0] == '1') 0 else 1
    for (i in 1 until s.length - 1) {
        if (s[i] == '1') {
            if (c % 2 == 1) {
                c++
                ones += i
            }
        } else {
            if (c % 2 == 0) {
                c++
                ones += i
            }
        }
    }

    return Math.min(ones, zero)
}

fun differenceOfDistinctValues(grid: Array<IntArray>): Array<IntArray> {

    val ans = Array(grid.size) { IntArray(grid[0].size) { 0 } }
    for (i in 0 until grid.size) {
        for (j in 0 until grid[i].size) {
            val l = topLeft(i, j, grid)
            val r = btmRight(i, j, grid)
            ans[i][j] = Math.abs(l - r)
        }
    }

    return ans
}

fun btmRight(i: Int, j: Int, grid: Array<IntArray>): Int {
    val set = mutableSetOf<Int>()
    var r = i + 1
    var c = j + 1
    while (r < grid.size && c < grid[0].size) {
        set.add(grid[r][c])
        r++
        c++
    }
    return set.size
}

fun topLeft(i: Int, j: Int, grid: Array<IntArray>): Int {
    val set = mutableSetOf<Int>()
    var r = i - 1
    var c = j - 1
    while (r >= 0 && c >= 0) {
        set.add(grid[r][c])
        r--
        c--
    }
    return set.size
}

fun removeTrailingZeros(num: String): String {
    var end = num.length - 1
    while (end >= 0 && num[end] == '0') {
        end--
    }
    if (end == -1) return ""
    return num.substring(0, end + 1)
}

fun longestRepeating(s: String, queryCharacters: String, queryIndices: IntArray): IntArray {


    return queryIndices
}

class SGTree(val str: String) {

    var root = Tree(0)
    var size = str.length

    init {
        initTree(0, 0, size - 1, root, str)
    }

    private fun initTree(at: Int, l: Int, r: Int, root: Tree, str: String) {
        println(
            longestRepeating(
                "babacc", "bcb", intArrayOf(
                    1, 3, 3
                )
            )
        )

        val obj = SGTree("bbbbbckkk")
        obj.dfs(obj.root)
        if (l == r) {
            root.curMax = 1
            root.prefixLen = 1
            root.suffixLen = 1
            return
        }
        if (root.left == null) {
            root.left = Tree(at * 2 + 1)
            root.right = Tree(at * 2 + 2)
        }
        val mid = (l + r) / 2
        initTree(at * 2 + 1, l, mid, root.left!!, str)
        initTree(at * 2 + 2, mid + 1, r, root.right!!, str)
        if (str[mid] == str[mid + 1]) {
            root.curMax = root.left!!.suffixLen + root.right!!.prefixLen
            root.prefixLen = root.curMax
            root.suffixLen = root.curMax
        } else {
            root.prefixLen = root.left!!.suffixLen
            root.suffixLen = root.right!!.prefixLen
        }
        root.curMax = Math.max(root.curMax, Math.max(root.prefixLen, root.suffixLen))
    }


    data class Tree(val at: Int) {
        var left: Tree? = null
        var right: Tree? = null
        var curMax = 0
        var prefixLen = 0
        var suffixLen = 0
    }

    fun dfs(root: Tree? = null) {
        if (root == null) return
        println("At : ${root.at} , Max : ${root.curMax} Prefix : ${root.prefixLen} Suffix : ${root.suffixLen}")
        dfs(root.left)
        dfs(root.right)
    }

}

fun kthPalindrome(queries: IntArray, intLength: Int): LongArray {

    val ans = LongArray(queries.size) { -1L }
    if (intLength == 1) {
        for (i in 0 until queries.size) {
            if (queries[i] <= 9) ans[i] = queries[i].toLong()
        }
        return ans
    }
    var power: Int
    if (intLength % 2 == 0) {
        power = Math.pow(10.0, (intLength / 2) - 1.0).toInt()
    } else {
        power = Math.pow(10.0, (intLength / 2).toDouble()).toInt()
    }
    val len = "$power".length
    for (i in 0 until queries.size) {
        val cur = queries[i]
        var min = -1L
        if (intLength % 2 == 1) {
            val new = power + (cur - 1)
            if ("$new".length == len) {
                val str = "$new"
                min = (str + str.substring(0, str.length - 1).reversed()).toLong()
            }
        } else {
            val new = power + (cur - 1)
            if ("$new".length == len) {
                min = ("$new" + "$new".reversed()).toLong()
            }
        }
        ans[i] = min
    }

    return ans
}

fun canTraverseAllPairs(nums: IntArray): Boolean {
    if (nums.contains(1)) return false
    val adjPrimes = hashMapOf<Int, MutableSet<Int>>()
    var start = -1
    for (n in nums) {
        val primes = getPrimes(n)
        if (primes.size == 1) {
            if (adjPrimes[primes[0]] == null) adjPrimes[primes[0]] = mutableSetOf()
        }
        for (i in 0 until primes.size) {
            start = primes[i]
            for (j in i + 1 until primes.size) {
                val u = primes[i]
                val v = primes[j]
                if (adjPrimes[u] == null) {
                    adjPrimes[u] = mutableSetOf()
                }
                if (adjPrimes[v] == null) {
                    adjPrimes[v] = mutableSetOf()
                }
                adjPrimes[u]!!.add(v)
                adjPrimes[v]!!.add(u)
            }
        }
    }
    if (start == -1) return false
    val set = mutableSetOf<Int>()
    dfs(start, set, adjPrimes)
    return set.size == adjPrimes.size
}

fun dfs(at: Int, visited: MutableSet<Int>, adj: HashMap<Int, MutableSet<Int>>) {
    visited.add(at)
    for (n in adj[at] ?: mutableSetOf()) {
        if (!visited.contains(n)) {
            dfs(n, visited, adj)
        }
    }
}

fun getPrimes(num: Int): List<Int> {
    if (isPrime(num)) return listOf(num)
    var cur = num
    val primes = mutableSetOf<Int>()
    var div = 2
    while (!isPrime(cur)) {
        if (cur % div == 0) {
            primes.add(div)
            cur /= div
        } else div++
        if (cur == 1) break
    }
    if (cur != 1) primes.add(cur)
    return primes.toList()
}

fun minExtraChar(s: String, dictionary: Array<String>): Int {
    val memo = hashMapOf<String, Int>()
    return dfs(s, dictionary, memo)
}


fun dfs(cur: String, dict: Array<String>, memo: HashMap<String, Int>): Int {
    if (memo.containsKey(cur)) return memo[cur]!!
    var min = cur.length
    for (d in dict) {
        val at = cur.indexOf(d)
        if (at != -1) {
            val begin = cur.substring(0, at)
            val end = cur.substring(at + d.length)
            val res = dfs(begin + end, dict, memo)
            min = Math.min(min, res)
        }
    }
    memo[cur] = min
    return min
}

fun buyChoco(prices: IntArray, money: Int): Int {
    prices.sort()
    val sum = prices[0] + prices[1]
    if (sum <= money) return money - sum
    return money
}

var maxQuery = 0L

var c = 0

fun dfs(l: Int, r: Int, board: IntArray, map: HashMap<Int, Long>) {
    if (l > r) {
        c++
        if (map.containsKey(c)) map[c] = board.joinToString("").toLong()
        return
    }
    for (i in 0..9) {
        board[l] = i
        board[r] = i
        dfs(l + 1, r - 1, board, map)
    }
}

fun kmp(pattern: String, text: String) {
    kmp("aba", "absdlwedababa")

}

fun sumScores(s: String): Long {

    var prefix = 0L
    var suffix = 0L
    var power = 1L
    var ans = 0L
    for (i in 0 until s.length) {
        val curPre = s[i] - '0'
        val curSuf = (s[s.length - 1 - i] - '0') * power
        prefix += curPre
        suffix += curSuf
        prefix %= mod
        suffix %= mod
        prefix *= 10
        power *= 10
        power %= mod
        prefix %= mod
    }

    return ans
}

fun maximumBeauty(flowers: IntArray, newFlowers: Long, target: Int, full: Int, partial: Int): Long {
    println(
        maximumBeauty(
            intArrayOf(1, 3, 1, 1), 7, 6, 12, 1
        )
    )

    return 0L
}

var hasFound = false

fun solve(at: Int, amount: Int, curFreq: IntArray, freq: IntArray) {
    if (hasFound || at < 0) return
    if (amount == 0) {
        hasFound = true
        println(curFreq.toList())
        return
    }
    if (moneys[at] > amount) solve(at - 1, amount, curFreq, freq)
    val div = amount / moneys[at]
    val remove = Math.min(div, freq[at]) * moneys[at]
    curFreq[at] += Math.min(div, freq[at])
    solve(at - 1, amount - remove, curFreq, freq)
}

val moneys = listOf(20, 50, 100, 200, 500)

class ATM() {

    val freq = hashMapOf<Int, Long>()

    init {
        for (i in 0 until moneys.size) {
            freq[i] = 0
        }
    }

    fun deposit(banknotesCount: IntArray) {
        for (i in 0 until banknotesCount.size) {
            freq[i] = freq.getOrDefault(i, 0) + banknotesCount[i]
        }
    }

    fun withdraw(amount: Int): IntArray {
        if (amount < 20) return intArrayOf(-1)
        if (amount % 10 != 0) return intArrayOf(-1)
        val clone = freq.clone() as HashMap<Int, Long>
        val ans = IntArray(freq.size) { 0 }
        var target = amount.toLong()
        var at = -1
        for (i in moneys.size - 1 downTo 0) {
            val cur = moneys[i]
            if (cur <= amount && clone.getOrDefault(i, 0) > 0) {
                target -= cur
                ans[i] = 1
                at = i
                break
            }
        }

        if (at == -1) return intArrayOf(-1)
        if (target == 0L) {
            freq[at] = freq.getOrDefault(at, 0) - 1
            return ans
        }
        clone[at] = clone.getOrDefault(at, 0) - 1
        for (i in at downTo 0) {
            if (clone[i] == 0L || target == 0L) continue
            val div = target / moneys[i]
            val remove = Math.min(div, clone.getOrDefault(i, 0)) * moneys[i]
            ans[i] += Math.min(div, clone.getOrDefault(i, 0)).toInt()
            target -= remove
        }
        if (target != 0L) return intArrayOf(-1)
        for (i in 0 until ans.size) {
            freq[i] = freq.getOrDefault(i, 0) - ans[i]
        }
        return ans
    }
    //20,50,100,200,500

}

fun maximumMinutes(grid: Array<IntArray>): Int {
    println(
        maximumMinutes(
            arrayOf(
                intArrayOf()
            )
        )
    )
    var l = 0
    var r = Math.pow(10.0, 9.0).toInt()
    if (isPossible(r, grid)) return r

    var ans = -1
    while (l <= r) {
        val mid = (l + r) / 2
        if (isPossible(mid, grid)) {
            ans = Math.max(ans, mid)
            l = mid + 1
        } else {
            r = mid - 1
        }
    }

    return ans
}

fun isPossible(stay: Int, grid: Array<IntArray>): Boolean {

    val state = grid.map { it.clone() }.toTypedArray().clone()
    val fires = LinkedList<IntArray>()
    for (i in 0 until state.size) {
        for (j in 0 until state[i].size) {
            if (state[i][j] == 1) fires.add(intArrayOf(i, j))
        }
    }
    var st = stay
    while (st >= 0) {
        val size = fires.size
        for (i in 0 until size) {
            val cur = fires.poll()
            for (dir in dirs) {
                val cI = cur[0] + dir[0]
                val cJ = cur[1] + dir[1]
                if (cI < 0 || cJ < 0 || cI == state.size || cJ == state[0].size || state[cI][cJ] == 2 || state[cI][cJ] == 1) continue
                fires.add(intArrayOf(cI, cJ))
                state[cI][cJ] = 1
            }
        }
        if (fires.isEmpty()) break
        st--
    }

    val person = LinkedList<IntArray>()
    if (state[0][0] == 1) return false
    person.add(intArrayOf(0, 0))
    var recentVisited = false
    while (person.isNotEmpty()) {
        var size = person.size
        recentVisited = false
        for (i in 0 until size) {
            val cur = person.poll()
            for (dir in dirs) {
                val cI = cur[0] + dir[0]
                val cJ = cur[1] + dir[1]
                if (cI == state.size - 1 && cJ == state[0].size - 1) {
                    if (recentVisited) return true
                    return state[cI][cJ] != 2 && state[cI][cJ] == 1
                }
                if (cI < 0 || cJ < 0 || cI == state.size || cJ == state[0].size || state[cI][cJ] == 2 || state[cI][cJ] == 1) continue
                person.add(intArrayOf(cI, cJ))
                state[cI][cJ] = 1
            }
        }
        size = fires.size
        for (i in 0 until size) {
            val cur = fires.poll()
            for (dir in dirs) {
                val cI = cur[0] + dir[0]
                val cJ = cur[1] + dir[1]
                if (cI < 0 || cJ < 0 || cI == state.size || cJ == state[0].size || state[cI][cJ] == 2 || state[cI][cJ] == 1) continue
                fires.add(intArrayOf(cI, cJ))
                state[cI][cJ] = 1
                if (cI == grid.size - 1 && cJ == grid[0].size - 1) recentVisited = true
            }
        }
    }
    return false
}


fun hitBricks(grid: Array<IntArray>, hits: Array<IntArray>): IntArray {
    println(
        hitBricks(
            arrayOf(
                intArrayOf(1, 1, 0, 0), intArrayOf(1, 1, 0, 0), intArrayOf(1, 1, 0, 0), intArrayOf(1, 1, 0, 0)
            ), arrayOf(
                intArrayOf(1, 1), intArrayOf(1, 0)
            )
        )
    )
    val stable = Array(grid.size) { BooleanArray(grid[0].size) { false } }
    for ((i, j) in hits) {
        grid[i][j] = 0
    }
    val bfs = LinkedList<IntArray>()
    for (c in 0 until grid[0].size) {
        if (grid[0][c] == 1) {
            bfs.add(intArrayOf(0, c))
            stable[0][c] = true
        }
    }
    while (bfs.isNotEmpty()) {
        val poll = bfs.poll()
        stable[poll[0]][poll[1]] = true
        for (dir in dirs) {
            val nI = dir[0] + poll[0]
            val nJ = dir[1] + poll[1]
            if (nI < 0 || nJ < 0 || nI == grid.size || nJ == grid[0].size || grid[nI][nJ] == 0 || stable[nI][nJ]) continue
            bfs.add(intArrayOf(nI, nJ))
            stable[nI][nJ] = true
        }
    }
    for (i in 0 until grid.size) {
        for (j in 0 until grid[i].size) {
            if (grid[i][j] == 1 && !stable[i][j]) {
                grid[i][j] = 0
            }
        }
    }

    val ans = IntArray(hits.size) { 0 }
    for (i in hits.size - 1 downTo 0) {
        val cur = hits[i]
        var neighStable = false
        for (dir in dirs) {
            val nI = cur[0] + dir[0]
            val nJ = cur[1] + dir[1]
            if (nI < 0 || nJ < 0 || nI == grid.size || nJ == grid[0].size || grid[nI][nJ] == 0) continue
            if (stable[nI][nJ]) {
                neighStable = true
                break
            }
        }
        var c = 0
        if (neighStable) {
            for (dir in dirs) {
                val nI = cur[0] + dir[0]
                val nJ = cur[1] + dir[1]
                if (nI < 0 || nJ < 0 || nI == grid.size || nJ == grid[0].size || grid[nI][nJ] == 0) continue
                c += dfs(nI, nJ, grid, stable)
            }
        } else {
            grid[cur[0]][cur[1]] = 1
        }
        println("C : $c")
        ans[i] = c
    }

    return intArrayOf()
}

fun dfs(i: Int, j: Int, grid: Array<IntArray>, stable: Array<BooleanArray>): Int {
    if (i < 0 || j < 0 || i == grid.size || j == grid[i].size || grid[i][j] == 0 || stable[i][j]) return 0
    var c = 0
    stable[i][j] = true
    c += dfs(i - 1, j, grid, stable)
    c += dfs(i, j - 1, grid, stable)
    c += dfs(i, j + 1, grid, stable)
    c += dfs(i + 1, j, grid, stable)
    return c
}

fun maximumScore(scores: IntArray, edges: Array<IntArray>): Int {
    println(
        maximumScore(
            intArrayOf(5, 2, 9, 8, 4), arrayOf(
                intArrayOf(0, 1),
                intArrayOf(1, 2),
                intArrayOf(2, 3),
                intArrayOf(0, 2),
                intArrayOf(1, 3),
                intArrayOf(2, 4)
            )
        )
    )
    val adj = Array(scores.size) { mutableListOf<Int>() }
    for ((from, to) in edges) {
        adj[from].add(to)
        adj[to].add(from)
    }
    val visited = BooleanArray(scores.size) { false }
    val memo = mutableSetOf<String>()
    for (i in 0 until scores.size) {
        visited[i] = true
        dfs(i, scores[i], 3, adj, scores, visited, memo)
        visited[i] = false
    }
    if (maxSeq == 0) return -1
    return maxSeq
}

var maxSeq = 0

fun dfs(
    at: Int,
    score: Int,
    soFar: Int,
    adj: Array<MutableList<Int>>,
    scores: IntArray,
    visited: BooleanArray,
    memo: MutableSet<String>
) {
    if (soFar == 0) {
        maxSeq = Math.max(maxSeq, score)
        return
    }
    val key = "$at|$score|$soFar"
    if (memo.contains(key)) return
    memo.add(key)
    val neighs = adj[at]
    for (n in neighs) {
        if (!visited[n]) {
            visited[n] = true
            dfs(n, score + scores[n], soFar - 1, adj, scores, visited, memo)
            visited[n] = false
        }
    }
}

fun uniqueLetterString(s: String): Int {

    println(
        appealSum(
            "abbca"
        )
    )

    println(
        uniqueLetterString(
            "abbca"
        )
    )

    return 0
}

fun appealSum(s: String): Long {

    var ans = 0L
    for (i in 1 until s.length) {
        ans += (countSubstringChars(s, i) * i)
    }

    return ans
}

fun countSubstringChars(str: String, distinct: Int): Int {

    val bigFreq = hashMapOf<Char, Int>()
    val smallFreq = hashMapOf<Char, Int>()
    var r = 1
    var mid = 1
    var ans = 0
    bigFreq[str[0]] = 1
    smallFreq[str[0]] = 1
    for (l in 0 until str.length) {

        while (r < str.length && (bigFreq.containsKey(str[r]) || bigFreq.size < distinct)) {
            bigFreq[str[r]] = bigFreq.getOrDefault(str[r], 0) + 1
            r++
        }
        while (mid < str.length && smallFreq.size != distinct) {
            val cur = str[mid]
            smallFreq[cur] = smallFreq.getOrDefault(cur, 0) + 1
            mid++
        }

        if (smallFreq.size == distinct) {
            ans += (r - mid + 1)
        }

        val cur = str[l]
        bigFreq[cur] = bigFreq.getOrDefault(cur, 0) - 1
        smallFreq[cur] = smallFreq.getOrDefault(cur, 0) - 1
        if (bigFreq[cur] == 0) bigFreq.remove(cur)
        if (smallFreq[cur] == 0) smallFreq.remove(cur)

    }

    return ans
}

fun maximumWhiteTiles(tiles: Array<IntArray>, carpetLen: Int): Int {

    var ans = 0
    var rAt = 0
    var lAt = 0
    tiles.sortWith(object : java.util.Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            return o1[0] - o2[0]
        }
    })
    var maxLen = 0
    for ((start, end) in tiles) {
        maxLen = Math.max(maxLen, end - start + 1)
        if (maxLen >= carpetLen) return carpetLen
    }
    val min = tiles[0][0] + carpetLen
    var sum = 0
    while (rAt < tiles.size) {
        if (tiles[rAt][0] > min) break
        val curTile = tiles[rAt]
        if (curTile[1] < min) {
            sum += (curTile[1] - curTile[0] + 1)
        } else {
            val cur = Math.min(curTile[1], min - 1) - curTile[0]
            sum += cur
        }
        if (curTile[1] < min) rAt++
        else break
    }
    println("Sum : $sum rAt : $rAt")
    ans = Math.max(ans, sum)
    val max = tiles[tiles.size - 1][1]
    for (r in min..max) {
        if (r in tiles[rAt][0]..tiles[rAt][1]) {
            sum++
        }
        if (r == tiles[rAt][1]) rAt++
        val l = r - carpetLen
        if (l >= 0) {
            if (l in tiles[lAt][0]..tiles[lAt][1]) {
                sum -= 1
            }
            if (l == tiles[lAt][1]) lAt++
        }
        ans = Math.max(ans, sum)
        if (rAt == tiles.size) break
    }

    return ans
}

fun distributeCookies(cookies: IntArray, k: Int): Int {
    println(
        distributeCookies(
            intArrayOf(8, 15, 10, 20, 8), 2
        )
    )
    return 0
}

fun dp(): Int {

    return 0
}

class CountIntervals() {

    val size = Math.pow(10.0, 9.0).toInt() + 1
    val obj = SolveTree(size)

    fun add(left: Int, right: Int) {
        val obj = CountIntervals()
        obj.add(2, 3)
        obj.add(7, 10)
//    println(obj.count())
        obj.add(5, 8)
        println(obj.count())
    }

    fun count(): Int {
        return obj.root.value
    }

}

class SolveTree(val size: Int) {

    val root = Tree()

    fun addRange(left: Int, right: Int) {
        addRangeHelper(root, 0, size - 1, left, right)
    }

    private fun addRangeHelper(root: Tree, l: Int, r: Int, qL: Int, qR: Int) {
        if (r < qL || qR < l) return
        else if (qL <= l && r <= qR) {
            val freq = r - l + 1
            root.value = freq
            return
        } else {
            val mid = (l + r) / 2
            if (root.left == null) {
                root.left = Tree()
                root.right = Tree()
            }
            addRangeHelper(root.left!!, l, mid, qL, qR)
            addRangeHelper(root.right!!, mid + 1, r, qL, qR)
            root.value = root.left!!.value + root.right!!.value
        }
    }

    class Tree {
        var value = 0
        var left: Tree? = null
        var right: Tree? = null
    }

}


fun countSubarrays(nums: IntArray, k: Long): Long {

    var ans = 0L
    var l = 0
    while (l < nums.size && nums[l] >= k) l++
    var sum = nums[l]
    var r = l
    while (l < nums.size) {
        if (sum == 0) {
            sum += nums[l]
            r = l
        }
        while (r < nums.size && sum * (r - l + 1) < k) {
            if (r + 1 < nums.size && (sum + nums[r + 1]) * (r - l + 2) < k) {
                r++
                sum += nums[r]
            } else break
        }
        if (sum * (r - l + 1) < k) ans += (r - l + 1)
        sum -= nums[l]
        l++
    }

    return ans
}


fun maxChunksToSorted(arr: IntArray): Int {

    val minFromEnd = IntArray(arr.size) { Int.MAX_VALUE }
    var min = arr[arr.size - 1]
    minFromEnd[minFromEnd.size - 1] = min
    for (i in arr.size - 2 downTo 0) {
        min = Math.min(min, arr[i])
        minFromEnd[i] = min
    }
    var ans = 0
    var at = 0
    var max = 0
    while (at < arr.size - 1) {
        max = Math.max(max, arr[at])
        var cur = arr[at]
        if (cur <= minFromEnd[at + 1]) {
            ans++
        } else if (cur > minFromEnd[at + 1]) {
            while (at + 1 < arr.size && cur > minFromEnd[at + 1]) {
                cur = Math.max(cur, arr[at])
                at++
            }
            ans++
        }
        at++
    }
    if (max <= minFromEnd[minFromEnd.size - 1]) ans++

    return ans
}

fun largestVariance(s: String): Int {

    val exist = BooleanArray(26) { false }
    println(
        largestVariance(
            "aababbb"
        )
    )

    return 0
}


class DTreeClone(val size: Int) {

    val root = DTree(0)

    fun addRange(left: Int, right: Int) {
        addRangeHelper(root, 0, size, left, right)
    }

    private fun addRangeHelper(root: DTree, l: Int, r: Int, qL: Int, qR: Int) {
        if (r < qL || qR < l) return
        else if (qL <= l && r <= qR) {
            val freq = r - l + 1
            if (freq != root.value) {
                root.value = freq
            }
            return
        } else {
            val mid = (l + r) / 2
            if (root.left == null) {
                root.left = DTree(root.at * 2 + 1)
                root.right = DTree(root.at * 2 + 2)
            }
            addRangeHelper(root.left!!, l, mid, qL, qR)
            addRangeHelper(root.right!!, mid + 1, r, qL, qR)
            root.value = root.left!!.value + root.right!!.value
        }
    }

    fun countFreq(): Int {
        return root.value
    }

    data class DTree(val at: Int) {
        var value: Int = 0
        var left: DTree? = null
        var right: DTree? = null
    }

}


fun totalSteps(nums: IntArray): Int {
    return recursion(nums.toList())
}

fun recursion(list: List<Int>): Int {

    val newList = mutableListOf<Int>()
    var at = 0
    var max = 0
    while (at < list.size) {
        if (at + 1 < list.size && list[at] <= list[at + 1]) {
            newList.add(list[at])
            at++
        } else if (at + 1 < list.size && list[at] > list[at + 1]) {
            var c = 0
            val cur = list[at]
            newList.add(cur)
            while (at < list.size && cur > list[at]) {
                at++
                c++
                if (at + 1 < list.size && list[at] > list[at + 1]) {
                    break
                }
            }
            max = Math.max(max, c)
        } else {
            newList.add(list[at])
            at++
        }
    }
    println(newList)

    return 0
}

class TextEditor() {

    var cursor = 0
    val board = StringBuilder()

    fun addText(text: String) {

    }

    fun deleteText(k: Int): Int {
        return 0
    }

    fun cursorLeft(k: Int): String {
        return ""
    }

    fun cursorRight(k: Int): String {
        return ""
    }

}

fun dp(bit: Int, cookies: IntArray, k: Int, target: Int): Int {

    if (bit == target) return 0

    var cur = 0
    var min = Int.MAX_VALUE
    for (i in 0 until cookies.size) {
        if (bit and (1 shl i) == 0) {
            val setBit = bit or (1 shl i)
            cur += cookies[i]
            val move = dp(setBit, cookies, k, target)
            val max = Math.max(cur, move)
            min = Math.min(min, max)
        }
    }

    return min
}

fun modifiedGraphEdges(
    n: Int, edges: Array<IntArray>, source: Int, destination: Int, target: Int
): Array<IntArray> {

    println(
        modifiedGraphEdges(
            5, arrayOf(
                intArrayOf(4, 1, -1), intArrayOf(2, 0, -1), intArrayOf(0, 3, -1), intArrayOf(4, 3, -1)
            ), source = 0, destination = 1, target = 5
        )
    )

    return arrayOf()
}

fun punishmentNumber(n: Int): Int {

    var c = 0
    for (i in 1..n) {
        val square = i * i
        if (isEqual(square, i)) {
            c += i
        }
    }

    return c
}

fun isEqual(cur: Int, target: Int): Boolean {
    val memo = hashMapOf<String, Boolean>()
    return dp(0, "$cur", target, memo)
}

fun dp(at: Int, number: String, target: Int, memo: HashMap<String, Boolean>): Boolean {
    if (at == number.length && target == 0) return true
    else if (target < 0) return false
    val key = "$at|$target"
    if (memo.containsKey(key)) return memo[key]!!
    var possible = false
    var cur = 0
    for (i in at until number.length) {
        cur = cur * 10 + (number[i] - '0')
        val res = dp(i + 1, number, target - cur, memo)
        if (res) {
            possible = true
            break
        }
    }
    memo[key] = possible
    return possible
}

fun minLength(s: String): Int {

    val stack = Stack<Char>()
    //AB //CD
    for (c in s) {
        var pop = false
        if (stack.isNotEmpty() && ((c == 'B' && stack.peek() == 'A') || (c == 'D' && stack.peek() == 'C'))) {
            pop = true
            stack.pop()
        }
        if (!pop) stack.push(c)
    }

    return stack.size
}


fun minimumNumbers(num: Int, k: Int): Int {
    val memo = IntArray(num + 1) { -1 }
    val res = dpMin(num, k, memo)
    if (res == Int.MAX_VALUE) return -1
    return res
}

fun dpMin(cur: Int, k: Int, memo: IntArray): Int {
    if ("$cur"["$cur".length - 1] - '0' == k) return 0
    if (cur == 0) return 0
    else if (cur < 0) return Int.MAX_VALUE
    if (memo[cur] != -1) return memo[cur]
    var min = Int.MAX_VALUE
    val queue = LinkedList<String>()
    queue.add("$k")
    while (queue.isNotEmpty()) {
        val poll = queue.poll().toInt()
        if (poll != 0) {
            val res = dpMin(cur - poll, k, memo)
            if (res != Int.MAX_VALUE) min = Math.min(min, res + 1)
        }
        for (i in 0..9) {
            val next = "$i$poll"
            if (next.toInt() != poll && next.toInt() <= cur) queue.add(next)
        }
    }
    memo[cur] = min
    return min
}

fun sellingWood(m: Int, n: Int, prices: Array<IntArray>): Long {
    prices.sortWith(object : java.util.Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            return o2[2] - o1[2]
        }
    })
    return dp(prices, m, n)
}

fun dp(prices: Array<IntArray>, row: Int, col: Int): Long {

    var max = 0L
    for (i in 0 until prices.size) {
        val (r, c, price) = prices[i]
        if (r <= row && c <= col) {
            val cur = price + dp(prices, row - r, col) + dp(prices, r, col - c)
            max = Math.max(max, cur)
        }
    }

    return max
}

fun distinctSequences(n: Int): Int {

    val gcd = Array(7) { IntArray(7) { 0 } }
    for (i in 1..6) {
        for (j in 1..6) {
            gcd[i][j] = findGCD(i, j)
        }
    }
    val memo = hashMapOf<String, Long>()

    return dp(1, 0, -1, n, gcd, memo).toInt()
}

fun dp(at: Int, prev1: Int, prev2: Int, max: Int, gcd: Array<IntArray>, memo: HashMap<String, Long>): Long {

    val key = "$at|$prev1|$prev2"
    if (memo.containsKey(key)) return memo[key]!!
    if (at > max) return 1L
    var res = 0L
    for (i in 1..6) {
        if (i != prev1 && i != prev2 && (prev2 == -1 || gcd[prev2][i] == 1)) {
            res += dp(at + 1, prev2, i, max, gcd, memo)
            res %= mod
        }
    }
    memo[key] = res

    return res
}

fun countHousePlacements(n: Int): Int {

    var memo = LongArray(n + 1) { -1L }
    var r1 = dpExploreHouse(1, n, memo)
    println("R1 : $r1")
    memo = LongArray(n + 1) { -1L }

    return dpExploreHouse(1, n, memo, r1).toInt()
}

fun dpExploreHouse(at: Int, n: Int, memo: LongArray, final: Long = 1L): Long {
    if (at > n) return final
    if (memo[at] != -1L) return memo[at]
    var cur = dpExploreHouse(at + 2, n, memo, final)
    cur += dpExploreHouse(at + 1, n, memo, final)
    cur %= mod
    memo[at] = cur
    return cur
}

fun minimumScore(nums: IntArray, edges: Array<IntArray>): Int {
    println(
        minimumScore(
            intArrayOf(
                1, 5, 5, 4, 11
            ), arrayOf(
                intArrayOf(0, 1), intArrayOf(1, 2), intArrayOf(1, 3), intArrayOf(3, 4)
            )
        )
    )
    return 0
}

fun peopleAwareOfSecret(n: Int, delay: Int, forget: Int): Int {
    val memo = LongArray(n + 1) { -1L }
    return (dpExploreSecret(1, delay, forget, n, memo) % mod).toInt()
}

fun dpExploreSecret(atDay: Int, delay: Int, forget: Int, n: Int, memo: LongArray): Long {

    if (memo[atDay] != -1L) return memo[atDay]
    var cur = 0L
    val max = atDay + forget
    for (i in atDay + delay until max) {
        if (i <= n) {
            cur += dpExploreSecret(i, delay, forget, n, memo)
            cur %= mod
        }
    }
    if (max > n) cur++
    memo[atDay] = cur

    return cur
}

fun validSubarraySize(nums: IntArray, threshold: Int): Int {
    println(
        validSubarraySize(
            nums = intArrayOf(3, 5), threshold = 5
        )
    )
    val treeMap = TreeMap<Int, RangeSubArr>()
    val map = Array(nums.size) { IntArray(2) { 0 } }
    for (i in 0 until nums.size) {
        val cur = intArrayOf(nums[i], i)
        map[i] = cur
    }
    map.sortWith(object : java.util.Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            return o2[0] - o1[0]
        }
    })

    for ((value, at) in map) {
        var add = false
        val floor = treeMap.floorKey(at - 1)
        if (floor != null && treeMap[floor]!!.r + 1 == at) {
            add = true
            val prev = treeMap[floor]!!
            val curRange = RangeSubArr(prev.l, at, prev.counter + 1)
            if (value > (threshold / curRange.counter)) return curRange.counter
            treeMap[floor] = curRange
        }
        val ceil = treeMap.ceilingKey(at + 1)
        if (ceil != null && treeMap[ceil]!!.l - 1 == at) {
            val prev = treeMap[ceil]!!
            if (add) {
                val prevFloor = treeMap[floor]!!
                treeMap[floor] = RangeSubArr(prevFloor.l, prev.r, prev.counter + prevFloor.counter)
                if (value > (threshold / prev.counter + prevFloor.counter)) return prev.counter + prevFloor.counter
                treeMap.remove(ceil)
            } else {
                treeMap[at] = RangeSubArr(at, prev.r, prev.counter + 1)
                treeMap.remove(ceil)
                if (value > (threshold / (prev.counter + 1))) return prev.counter + 1
            }
            add = true
        }
        if (!add) {
            treeMap[at] = RangeSubArr(at, at, 1)
            if (value > threshold) return 1
        }
        println(treeMap)
    }

    return -1
}

data class RangeSubArr(val l: Int, val r: Int, val counter: Int)

fun minSumSquareDiff(nums1: IntArray, nums2: IntArray, k1: Int, k2: Int): Long {
    println(
        minSumSquareDiff(
            intArrayOf(1, 4, 10, 12), intArrayOf(5, 8, 6, 9), 1, 1
        )
    )
    val difs = IntArray(nums1.size)
    for (i in 0 until nums1.size) {
        difs[i] = Math.abs(nums1[i] - nums2[i])
    }
    difs.sort()

    return 0
}

fun countPaths(grid: Array<IntArray>): Int {

    var ans = 0L
    val memo = Array(grid.size) { LongArray(grid[0].size) { -1 } }
    for (i in 0 until grid.size) {
        for (j in 0 until grid[i].size) {
            val cur = dfs(i, j, grid, memo)
            ans += cur
            ans %= mod
        }
    }

    return ans.toInt()
}

fun dfs(i: Int, j: Int, grid: Array<IntArray>, memo: Array<LongArray>): Long {
    if (i < 0 || j < 0 || i == grid.size || j == grid[i].size) return 0L
    if (memo[i][j] != -1L) {
        return memo[i][j]
    }
    var cur = 1L
    val value = grid[i][j]
    if (i + 1 < grid.size && grid[i + 1][j] > value) cur += dfs(i + 1, j, grid, memo)
    if (i - 1 >= 0 && grid[i - 1][j] > value) cur += dfs(i - 1, j, grid, memo)
    if (j + 1 < grid[i].size && grid[i][j + 1] > value) cur += dfs(i, j + 1, grid, memo)
    if (j - 1 >= 0 && grid[i][j - 1] > value) cur += dfs(i, j - 1, grid, memo)
    cur %= mod
    memo[i][j] = cur
    return cur
}

fun countExcellentPairs(nums: IntArray, k: Int): Long {
    println(
        countExcellentPairs(
            intArrayOf(3, 5), 2
        )
    )
    val sorted = nums.toSet().toList()
    val bitFreq = IntArray(32) { 0 }
    for (i in 0 until sorted.size) {
        var c = 0
        val bin = Integer.toBinaryString(sorted[i])
        for (j in bin) {
            if (j == '1') c++
        }
        bitFreq[c]++
    }
    println(bitFreq.toList())
    var ans = 0L
    for (i in 1 until bitFreq.size) {
        for (j in 1 until bitFreq.size) {
            if (i + j >= k) {
                ans += (bitFreq[i] * bitFreq[j])
            }
        }
    }

    return ans
}

fun idealArrays(n: Int, maxValue: Int): Int {
    var ans = 0L
    val memo = hashMapOf<String, Long>()
    for (i in 1..maxValue) {
        ans += dpExploreIdeal(i, n - 1, maxValue, memo) % mod
    }
    return ans.toInt()
}

fun dpExploreIdeal(cur: Int, len: Int, maxValue: Int, memo: HashMap<String, Long>): Long {
    if (len == 0) {
        return 1
    }
    val key = "$cur|$len"
    if (memo.containsKey(key)) return memo[key]!!
    var ans = dpExploreIdeal(cur, len - 1, maxValue, memo)
    var c = 2
    while (c * cur <= maxValue) {
        ans += dpExploreIdeal(c * cur, len - 1, maxValue, memo)
        c++
    }
    memo[key] = ans % mod
    return ans
}

fun shortestSequence(rolls: IntArray, k: Int): Int {
    println(
        shortestSequence(
            rolls = intArrayOf(4, 2, 1, 2, 3, 3, 2, 4, 1), k = 4
        )
    )
    return 0
}

fun countSetBit(n: Int): Int {
    var c = 0
    var cur = n
    while (cur != 1) {
        if (cur % 2 == 1) c++
        cur /= 2
    }
    return 0
}

fun minimumReplacement(nums: IntArray): Long {

    var ans = 0L
    var max = nums[nums.size - 1]
    for (i in nums.size - 2 downTo 0) {
        val cur = solveDiv(nums[i], max)
        ans += cur[0]
        max = cur[1]
    }

    return ans
}

//div,max
fun solveDiv(cur: Int, max: Int): IntArray {
    if (cur <= max) return intArrayOf(0, cur)
    if (cur % max == 0) return intArrayOf(cur / max - 1, max)
    val c = cur / max
    var rem = cur % max
    var counter = 0
    var min = max
    while (rem < max) {
        rem++
        counter++
        min = Math.min(min, max - counter)
        var i = c
        while (i > 0 && rem < max) {
            rem++
            i--
        }
    }
    return intArrayOf(c, min)
}

fun longestIdealString(s: String, k: Int): Int {

    val nums = IntArray(s.length) { 0 }
    for (i in 0 until s.length) {
        val n = s[i] - 'a'
        nums[i] = n
    }
    var ans = 1
    val lastLoc = IntArray(26) { 0 }
    for (i in 0 until s.length) {
        val n = nums[i]
        var max = 1
        val l = Math.max(0, n - k)
        val r = Math.min(25, n + k)
        for (j in l..r) {
            max = Math.max(max, lastLoc[j] + 1)
        }
        lastLoc[n] = max
        ans = Math.max(ans, max)
    }

    return ans
}

fun buildMatrix(k: Int, rowConditions: Array<IntArray>, colConditions: Array<IntArray>): Array<IntArray> {

    println(
        buildMatrix(
            3, arrayOf(
                intArrayOf(1, 2), intArrayOf(3, 2)
            ), arrayOf(
                intArrayOf(2, 1), intArrayOf(3, 2)
            )
        )
    )


    return arrayOf()
}


fun countSpecialNumbers(n: Int): Int {


    return 0
}

fun maximumSegmentSum(nums: IntArray, removeQueries: IntArray): LongArray {

    val removed = BooleanArray(nums.size) { false }
    for (r in removeQueries) removed[r] = true
    val treeMap = TreeMap<Int, Range>()
    var at = 0
    var max = 0L
    while (at < nums.size) {
        if (!removed[at]) {
            var sum = 0L
            val l = at
            while (at < nums.size && removed[at]) {
                sum += nums[at]
                at++
            }
            treeMap[at] = Range(l, at - 1, sum)
            max = Math.max(max, sum)
        } else at++
    }
    val ans = LongArray(removeQueries.size) { 0L }
    ans[ans.size - 1] = max
    for (i in ans.size - 1 downTo 1) {
        val rem = removeQueries[i]
        val floor = treeMap.floorKey(rem)
        var added = false
        if (floor != null && treeMap[floor]!!.r + 1 == rem) {
            val range = treeMap[floor]!!
            treeMap[floor] = Range(range.l, rem, range.sum + nums[rem])
            added = true
            max = Math.max(max, range.sum + nums[rem])
        }
        val ceil = treeMap.ceilingKey(rem)
        if (ceil != null && treeMap[ceil]!!.l - 1 == rem) {
            if (added) {
                val leftRange = treeMap[floor]!!
                val rightRange = treeMap[ceil]!!
                treeMap[floor] = Range(leftRange.l, rightRange.r, leftRange.sum + rightRange.sum)
                treeMap.remove(ceil)
                max = Math.max(max, leftRange.sum + rightRange.sum)
            } else {
                val range = treeMap[ceil]!!
                treeMap[ceil] = Range(rem, range.r, range.sum + nums[rem])
                max = Math.max(max, range.sum + nums[rem])
            }
            added = true
        }
        if (!added) {
            treeMap[rem] = Range(rem, rem, nums[rem] + 0L)
            max = Math.max(max, nums[rem] + 0L)
        }
        ans[i - 1] = max
    }

    return ans
}

data class Range(val l: Int, val r: Int, val sum: Long)

fun kSum(nums: IntArray, k: Int): Long {

    println(
        kSum(
            intArrayOf(1, -2, 3, 4, -10, 12), 16
        )
    )

    return 0L
}

fun mostBooked(n: Int, meetings: Array<IntArray>): Int {

    println(
        mostBooked(
            3, arrayOf(
                intArrayOf(0, 10),
                intArrayOf(1, 9),
                intArrayOf(2, 8),
                intArrayOf(3, 7),
                intArrayOf(4, 6),
            )
        )
    )
    val usedRoom = IntArray(n) { 0 }
    meetings.sortWith(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            return o1[0] - o2[0]
        }
    })

    val queue = PriorityQueue(object : Comparator<Booked> {
        override fun compare(o1: Booked, o2: Booked): Int {
            return o1.till - o2.till
        }
    })
    val freeRooms = TreeSet<Int>()
    for (f in 0 until n) {
        queue.add(Booked(0, f))
    }

    var at = 0
    var max = 0
    while (at < meetings.size) {
        val (start, duration) = meetings[at]
        while (queue.isNotEmpty() && queue.peek().till <= start) {
            freeRooms.add(queue.poll().room)
        }
        val freeRoom = freeRooms.firstOrNull()
        if (freeRoom == null) {
            meetings[at][0]++
        } else {
            queue.add(Booked(start + duration, freeRoom))
            freeRooms.remove(freeRoom)
            at++
            usedRoom[freeRoom]++
            max = Math.max(max, usedRoom[freeRoom])
        }
        println(queue)
    }

    println(usedRoom.toList())
    for (i in 0 until usedRoom.size) {
        if (usedRoom[i] == max) return i
    }

    return 0
}

data class Booked(val till: Int, val room: Int)

fun lengthOfLIS(nums: IntArray, k: Int): Int {

    var ans = 1
    println(
        lengthOfLIS(
            intArrayOf(7, 4, 5, 1, 8, 12, 4, 7), 5
        )
    )

    return ans
}

fun countDaysTogether(arriveAlice: String, leaveAlice: String, arriveBob: String, leaveBob: String): Int {

    val days = intArrayOf(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    val startDate = getStart(arriveAlice, arriveBob)
    val endDate = endDate(leaveAlice, leaveBob)
    if (startDate[0] >= endDate[0]) {
        if (startDate[0] > endDate[0]) return 0
        if (startDate[1] > endDate[1]) return 0
        return endDate[1] - startDate[1] + 1
    }
    var m = startDate[0] + 1
    var c = days[m] - startDate[1] + 1
    while (m < endDate[0]) {
        c += days[m]
        m++
    }
    c += endDate[1]

    return c
}

fun endDate(leaveAlice: String, leaveBob: String): List<Int> {
    val d1 = getMonthDate(leaveAlice)
    val d2 = getMonthDate(leaveBob)
    if (d1[0] == d2[0]) {
        if (d1[1] <= d2[1]) return d1
        return d2
    }
    if (d1[0] < d2[0]) return d1
    return d2
}

fun getStart(arriveAlice: String, arriveBob: String): List<Int> {
    val d1 = getMonthDate(arriveAlice)
    val d2 = getMonthDate(arriveBob)
    if (d1[0] == d2[0]) {
        if (d1[1] <= d2[1]) return d2
        return d1
    }
    if (d1[0] < d2[0]) return d2
    return d1
}

fun getMonthDate(str: String): List<Int> {
    return str.split("-").map { it.toInt() }
}

fun smallestSubarrays(nums: IntArray): IntArray {

    val bitLoc = hashMapOf<Int, Int>()
    val ans = IntArray(nums.size) { 0 }
    ans[ans.size - 1] = 1
    val binary = Integer.toBinaryString(nums[nums.size - 1]).reversed()
    for (i in 0 until binary.length) {
        if (binary[i] == '1') bitLoc[i] = nums.size - 1
    }
    for (i in nums.size - 2 downTo 0) {
        val bin = Integer.toBinaryString(nums[i]).reversed()
        var max = i
        for (bit in 0 until 32) {
            if (bit < bin.length) {
                if (bin[bit] == '0') {
                    max = Math.max(max, bitLoc.getOrDefault(bit, 0))
                }
            } else {
                max = Math.max(max, bitLoc.getOrDefault(bit, 0))
            }
        }
        ans[i] = max - i + 1
        for (j in 0 until bin.length) {
            if (bin[j] == '1') {
                bitLoc[j] = i
            }
        }
    }

    return ans
}

fun minimumMoney(transactions: Array<IntArray>): Long {
    println(
        minimumMoney(
            arrayOf(
                intArrayOf(2, 1), intArrayOf(5, 0), intArrayOf(4, 2)
            )
        )
    )
    val list = mutableListOf<Transaction>()
    for ((cost, cashBack) in transactions) {
        val tran: Transaction
        if (cost == cashBack) {
            tran = Transaction(cost, cashBack, 1)
        } else if (cost < cashBack) {
            tran = Transaction(cost, cashBack, 3)
        } else {
            tran = Transaction(cost, cashBack, 2)
        }
        list.add(tran)
    }
    val sorted = list.sortedWith(object : java.util.Comparator<Transaction> {
        override fun compare(o1: Transaction, o2: Transaction): Int {
            if (o1.important == o2.important) {
                return o2.cost - o1.cost
            }
            return o1.important - o2.important
        }
    })
    println(sorted)
    var l = 0L
    var r = Long.MAX_VALUE
    var ans = Long.MAX_VALUE
    while (l <= r) {
        val mid = l + (r - l) / 2
        if (isPossible(sorted, mid)) {
            ans = Math.min(ans, mid)
            r = mid - 1
        } else l = mid + 1
    }

    return ans
}

fun isPossible(sorted: List<Transaction>, mid: Long): Boolean {

    var money = mid
    for (tran in sorted) {
        if (money < tran.cost) return false
        money -= tran.cost
        money += tran.cashBack
    }

    return true
}

data class Transaction(val cost: Int, val cashBack: Int, val important: Int)

fun mincostToHireWorkers(quality: IntArray, wage: IntArray, k: Int): Double {

    val rations = DoubleArray(quality.size) { 0.0 }
    for (i in 0 until quality.size) {
        rations[i] = wage[i].toDouble() / quality[i].toDouble()
    }
    rations.sort()
    println(rations.toList())
    var l = 0
    var r = rations.size - 1
    var min = Double.MAX_VALUE
    while (l <= r) {
        val mid = (l + r) / 2
        val ratio = rations[mid]
        if (isPossible(ratio, k, quality, wage)) {
            min = Math.min(min, ratio)
            r = mid - 1
        } else l = mid + 1
    }
    println("Min : $min")
    val workers = mutableListOf<Double>()
    for (i in 0 until quality.size) {
        val needPay = quality[i] * min
        if (needPay >= wage[i].toDouble()) {
            workers.add(needPay)
        }
    }
    workers.sort()
    var ans = 0.0
    for (i in 0 until k) {
        ans += workers[i]
    }

    return ans
}

fun isPossible(ratio: Double, target: Int, quality: IntArray, wage: IntArray): Boolean {
    var c = 0
    for (i in 0 until quality.size) {
        val total = quality[i] * ratio
        if (total >= wage[i].toDouble()) c++
    }
    return c >= target
}


fun doesValidArrayExist(derived: IntArray): Boolean {

    if (derived[0] == 0) {
        if (isValid(0, 0, derived)) return true
        if (isValid(1, 1, derived)) return true
    } else {
        if (isValid(0, 1, derived)) return true
        if (isValid(1, 0, derived)) return true
    }
    return false
}

fun isValid(f: Int, s: Int, derived: IntArray): Boolean {

    val original = IntArray(derived.size) { 0 }
    original[0] = f
    original[1] = s
    original[original.size - 1] = original[0] xor derived[derived.size - 1]
    for (i in original.size - 2 downTo 2) {
        original[i] = original[i + 1] xor derived[i]
    }
    for (i in 0 until derived.size - 1) {
        if (original[i] xor original[i + 1] != derived[i]) return false
    }

    return original[original.size - 1] xor original[0] == derived[derived.size - 1]
}

fun countCompleteComponents(n: Int, edges: Array<IntArray>): Int {

    val adj = Array(n) { mutableListOf<Int>() }
    for ((from, to) in edges) {
        adj[from].add(to)
        adj[to].add(from)
    }
    var ans = 0
    val visited = BooleanArray(n) { false }
    for (i in 0 until visited.size) {
        if (!visited[i]) {
            val nodes = mutableListOf<Int>()
            dfs(i, visited, nodes, adj)
            var isValid = true
            for (j in nodes) {
                if (adj[j].size != nodes.size - 1) {
                    isValid = false
                }
            }
            println(nodes)
            if (isValid) ans++
        }
    }

    return ans
}

fun dfs(at: Int, visited: BooleanArray, nodes: MutableList<Int>, adj: Array<MutableList<Int>>) {
    if (visited[at]) return
    visited[at] = true
    nodes.add(at)
    for (n in adj[at]) {
        dfs(n, visited, nodes, adj)
    }
}

fun maxMoves(grid: Array<IntArray>): Int {
    val memo = Array(grid.size) { IntArray(grid[0].size) { -1 } }
    for (row in 0 until grid.size) {
        dpExploreMoves(row, 0, grid, memo)
    }
    var max = 0
    for (row in 0 until grid.size) {
        max = Math.max(max, memo[row][0] - 1)
    }
    return max
}

fun dpExploreMoves(i: Int, j: Int, grid: Array<IntArray>, memo: Array<IntArray>): Int {
    if (j == grid[0].size - 1) return 1
    if (memo[i][j] != -1) return memo[i][j]
    val cur = grid[i][j]
    var max = 1
    if (i - 1 >= 0 && grid[i - 1][j + 1] > cur) {
        max = Math.max(max, 1 + dpExploreMoves(i - 1, j + 1, grid, memo))
    }
    if (i + 1 < grid.size && grid[i + 1][j + 1] > cur) {
        max = Math.max(max, 1 + dpExploreMoves(i + 1, j + 1, grid, memo))
    }
    if (grid[i][j + 1] > cur) {
        max = Math.max(max, 1 + dpExploreMoves(i, j + 1, grid, memo))
    }
    memo[i][j] = max
    return max
}

fun circularGameLosers(n: Int, k: Int): IntArray {

    val visited = BooleanArray(n) { false }
    visited[0] = true
    var next = k % n
    var c = 1
    while (!visited[next]) {
        visited[next] = true
        c++
        next = (next + (c * k)) % (n)
    }
    val ans = mutableListOf<Int>()
    for (v in 0 until visited.size) {
        if (!visited[v]) ans.add(v + 1)
    }
    println(ans.toList())

    return ans.toIntArray()
}

fun maximumSum(arr: IntArray): Int {

    println(
        maximumSum(
            intArrayOf(1, -2, -2, 3)
        )
    )

    return 0
}

fun sumOfPower(nums: IntArray): Int {


    return 0
}

fun maximumOr(nums: IntArray, k: Int): Long {
    println(
        maximumOr(
            intArrayOf(8, 1, 2), 2
        )
    )
    var maxLeftBit = 0
    val list = mutableListOf<Int>()
    for (n in nums) {
        val bit = Integer.toBinaryString(n)
        if (bit.length > maxLeftBit) {
            list.clear()
            list.add(n)
            maxLeftBit = bit.length
        } else if (bit.length == maxLeftBit) {
            list.add(n)
        }
    }
    var ans = 0L
    for (n in list) {
        var cur = k
        var num = n.toLong()
        while (cur > 0) {
            num *= 2
            cur--
        }
        var xor = num
        var firstTime = true
        for (i in nums) {
            if (i == n && firstTime) {
                firstTime = false
                continue
            }
            xor = xor xor i.toLong()
        }
        ans = Math.max(ans, xor)
    }

    return ans
}

fun matrixSum(nums: Array<IntArray>): Int {
    nums.forEach { it.sort() }
    var score = 0
    for (col in nums[0].size - 1 downTo 0) {
        var max = 0
        for (row in 0 until nums.size) {
            max = Math.max(max, nums[row][col])
        }
        score += max
    }
    return score
}

fun countSeniors(details: Array<String>): Int {
    var c = 0
    for (de in details) {
        val age = de.substring(12, 14)
        if (age.toInt() >= 60) c++
    }
    return c
}

fun threeEqualParts(arr: IntArray): IntArray {

    var ones = 0
    for (n in arr) {
        if (n == 1) ones++
    }
    if (ones % 3 != 0) return intArrayOf(-1, -1)
    if (ones == 0) return intArrayOf(0, 2)
    var zeroEnd = 0
    val div = ones / 3
    var end = arr.size - 1
    while (end >= 0 && arr[end] == 0) {
        zeroEnd++
        end--
    }
    println("End Ones Starts : $end")
    var c = 0
    while (c < div) {
        if (arr[end] == 1) c++
        end--
    }
    ++end
    println("J pos : $end")
    var start = 0
    while (start < arr.size && arr[start] == 0) start++
    var i = start
    var j = end
    while (j < arr.size && arr[i] == arr[j]) {
        i++
        j++
    }
    println("I : $i and J : $j")
    if (j != arr.size) return intArrayOf(-1, -1)
    val ans = intArrayOf(-1, -1)
    ans[0] = i - 1
    ans[1] = end
    while (i < ans[1] && arr[i] == 0) {
        i++
    }
    j = ans[1]

    while (i < end && j < arr.size && arr[i] == arr[j]) {
        i++
        j++
    }
    if (i != end) return intArrayOf(-1, -1)
    return ans
}

fun largestComponentSize(nums: IntArray): Int {

    val adj = hashMapOf<Int, MutableSet<Int>>()
    val factors = hashMapOf<Int, MutableList<Int>>()
    for (f in nums) {
        val divisors = getDivisors(f)
        for (div in divisors) {
            if (factors[div] == null) factors[div] = mutableListOf()
            factors[div]!!.add(f)
        }
        for (i in 0 until divisors.size) {
            for (j in i + 1 until divisors.size) {
                val a = divisors[i]
                val b = divisors[j]
                if (adj[a] == null) adj[a] = mutableSetOf()
                if (adj[b] == null) adj[b] = mutableSetOf()
                adj[a]!!.add(b)
                adj[b]!!.add(a)
            }
        }
    }
    var ans = 1
    val visited = mutableSetOf<Int>()
    for ((node, neighs) in adj) {
        if (visited.contains(node)) continue
        val size = mutableSetOf<Int>()
        val seen = mutableSetOf<Int>()
        dfs(size, node, adj, factors, seen)
        println("Size : $size")
        ans = Math.max(ans, size.size)
    }

    return ans
}

fun dfs(
    curCompoSize: MutableSet<Int>,
    at: Int,
    adj: java.util.HashMap<Int, MutableSet<Int>>,
    factors: java.util.HashMap<Int, MutableList<Int>>,
    seen: MutableSet<Int>
) {
    val facts = factors[at]!!
    seen.add(at)
    curCompoSize.addAll(facts)
    for (n in adj[at] ?: return) {
        if (!seen.contains(n)) {
            dfs(curCompoSize, n, adj, factors, seen)
        }
    }
}

fun getDivisors(n: Int): List<Int> {
    if (isPrime(n)) return listOf(n)
    val divisors = mutableSetOf<Int>()
    var cur = n
    var counter = 2
    while (cur != 1) {
        if (cur % counter == 0) {
            cur /= counter
            divisors.add(counter)
        } else counter++
    }
    return divisors.toList()
}

fun xorAllNums(nums1: IntArray, nums2: IntArray): Int {

    println(
        xorAllNums(
            intArrayOf(2, 1, 3), intArrayOf(10, 2, 5, 0)
        )
    )

    return 0
}

fun deleteString(s: String): Int {
    val memo = IntArray(s.length) { -1 }
    return dpExplore(0, s, memo)
}

fun dpExplore(at: Int, s: String, memo: IntArray): Int {
    if (at >= s.length) return 0
    if (at + 1 == s.length - 1) {
        if (s[at] == s[at + 1]) return 2
        return 1
    }
    if (memo[at] != -1) return memo[at]
    val till = at + (s.length - at) / 2
    var ans = 1
    for (i in at until till) {
        if (isEqual(at, i, s)) {
            val cur = 1 + dpExplore(i + 1, s, memo)
            ans = Math.max(ans, cur)
        }
    }
    memo[at] = ans
    return ans
}

fun isEqual(from: Int, to: Int, s: String): Boolean {
    var h1 = 0L
    var h2 = 0L
    var power = 1L
    val dist = to - from + 1
    for (i in from..to) {
        h1 += (s[i] - 'a' + 1) * power
        h2 += (s[i + dist] - 'a' + 1) * power
        power *= 26
        power %= mod
        h1 %= mod
        h2 %= mod
    }
    return h1 == h2
}

//Subsequence With the Minimum Score
fun minimumScore(s: String, t: String): Int {

    println(
        minimumScore(
            "abacaba", "bzaa"
        )
    )

    return 0
}

fun rearrangeSticks(n: Int, k: Int): Int {
    return dpExplore(k, 1, n).toInt()
}

fun dpExplore(visible: Int, at: Int, n: Int): Long {
    if (visible == 0) return 1L
    if (at > n) return 0L
    var ways = dpExplore(visible, at + 1, n)
    ways += dpExplore(visible - 1, at + 1, n)
    return ways % mod
}

fun componentValue(nums: IntArray, edges: Array<IntArray>): Int {

    println("vzhofnpo".toCharArray().toList().sorted())
    println(
        componentValue(
            intArrayOf(1, 2, 2, 1, 1), arrayOf(
                intArrayOf(0, 2), intArrayOf(0, 4), intArrayOf(1, 4), intArrayOf(2, 3)
            )
        )
    )

    return 0
}

fun dfs(nodes: MutableSet<Int>, at: Int, value: Int, adj: Array<MutableList<Int>>, values: IntArray): Boolean {
    if (nodes.contains(at)) return false
    nodes.add(at)
    var isNotValid = false
    for (n in adj[at]) {
        if (values[n] != value) {
            isNotValid = true
        }
        isNotValid = isNotValid || dfs(nodes, n, value, adj, values)
    }
    return isNotValid
}

fun dfs(visited: BooleanArray, at: Int, value: Int, adj: Array<MutableList<Int>>, values: IntArray): Int {
    if (visited[at]) return 0
    visited[at] = true
    var cur = 0
    for (n in adj[at]) {
        if (!visited[n] && values[n] == value) {
            cur++
            cur += dfs(visited, n, value, adj, values)
        }
    }
    return cur
}

fun makeSimilar(nums: IntArray, target: IntArray): Long {

    println(
        makeSimilar(
            intArrayOf(1, 2, 5), intArrayOf(4, 1, 3)
        )
    )

    val odds = mutableListOf<Int>()
    val evens = mutableListOf<Int>()
    val targetOdds = mutableListOf<Int>()
    val targetEvens = mutableListOf<Int>()
    for (n in nums) {
        if (n % 2 == 0) {
            evens.add(n)
        } else {
            odds.add(n)
        }
    }
    for (n in target) {
        if (n % 2 == 0) {
            targetEvens.add(n)
        } else {
            targetOdds.add(n)
        }
    }

    odds.sort()
    evens.sort()
    targetOdds.sort()
    targetEvens.sort()

    var c = 0
    var ans = 0L
    for (i in 0 until evens.size) {
        val cur = evens[i]
        val tCur = targetEvens[i]
        if (cur == tCur) continue
        if (cur > tCur) {
            val abs = cur - tCur
            if (c <= 0) {
                ans += (abs / 2)
            } else {
                if (c - abs < 0) {
                    ans += (Math.abs(c - abs)) / 2
                }
            }
            c -= abs
        } else {
            val abs = tCur - cur
            if (c >= 0) ans += (abs / 2)
            else {
                if (c + abs > 0) {
                    ans += (c + abs) / 2
                }
            }
            c += abs
        }
    }

    for (i in 0 until odds.size) {
        val cur = odds[i]
        val tCur = targetOdds[i]
        if (cur == tCur) continue
        if (cur > tCur) {
            val abs = cur - tCur
            if (c <= 0) {
                ans += (abs / 2)
            } else {
                if (c - abs < 0) {
                    ans += (Math.abs(c - abs)) / 2
                }
            }
            c -= abs
        } else {
            val abs = tCur - cur
            if (c >= 0) ans += (abs / 2)
            else {
                if (c + abs > 0) {
                    ans += (c + abs) / 2
                }
            }
            c += abs
        }
    }

    return ans
}

fun secondGreaterElement(nums: IntArray): IntArray {
    println(
        secondGreaterElement(
            intArrayOf(2, 4, 0, 9, 6)
        )
    )
    return nums
}

fun countPalindromes(s: String): Int {
    println(
        countPalindromes(
            "0000000"
        )
    )
    var ans = 0L
    for (i in 2 until s.length - 2) {
        val cur = dpExplore(i - 1, i + 1, 0, s) % mod
        println("Cur : $cur, Centre : $i")
        ans += cur
    }

    return (ans % mod).toInt()
}

fun dpExplore(l: Int, r: Int, c: Int, s: String): Long {
    if (c == 2) return 1L
    if (l < 0 || r == s.length) return 0
    var cur = 0L
    if (s[l] == s[r]) {
        cur += dpExplore(l - 1, r + 1, c + 1, s)
    }
    cur += dpExplore(l - 1, r, c, s)
    cur += dpExplore(l, r + 1, c, s)
    return cur % mod
}

fun magnificentSets(n: Int, edges: Array<IntArray>): Int {

    val indegrees = IntArray(n + 1) { 0 }
    val adj = Array(n + 1) { mutableListOf<Int>() }
    for ((from, to) in edges) {
        adj[from].add(to)
        adj[to].add(from)
        indegrees[from]++
        indegrees[to]++
    }
    if (!isGraphBipartite(n, adj)) return -1
    val visited = BooleanArray(n + 1) { false }
    var groups = 0
    for (i in 1 until visited.size) {
        if (!visited[i]) {
            val nodes = mutableSetOf<Int>()
            dfs(i, nodes, adj)
            var maxDepth = 0
            nodes.forEach {
                visited[it] = true
                maxDepth = Math.max(maxDepth, bfs(it, adj))
            }
            groups += maxDepth
        }
    }
    return groups
}

fun bfs(at: Int, adj: Array<MutableList<Int>>): Int {
    val visited = BooleanArray(adj.size) { false }
    val queue = LinkedList<Int>()
    var level = 0
    queue.add(at)
    visited[at] = true
    while (queue.isNotEmpty()) {
        val size = queue.size
        level++
        for (i in 0 until size) {
            val poll = queue.poll()
            for (n in adj[poll]) {
                if (!visited[n]) {
                    queue.add(n)
                    visited[n] = true
                }
            }
        }
    }
    return level
}

fun dfs(at: Int, nodes: MutableSet<Int>, adj: Array<MutableList<Int>>): Int {
    if (nodes.contains(at)) return 0
    var cur = 1
    nodes.add(at)
    for (n in adj[at]) {
        cur += dfs(n, nodes, adj)
    }
    return cur
}

fun isGraphBipartite(n: Int, adj: Array<MutableList<Int>>): Boolean {
    val groups = IntArray(n + 1) { -1 }
    for (i in 1 until groups.size) {
        if (groups[i] == -1 && !canBeDividedIntoTwo(groups, i, 0, adj)) return false
    }
    return true
}

fun canBeDividedIntoTwo(groups: IntArray, at: Int, group: Int, adj: Array<MutableList<Int>>): Boolean {
    if (groups[at] != -1) {
        return groups[at] == group
    }
    groups[at] = group
    for (n in adj[at]) {
        val nextGroup = if (group == 0) 1 else 0
        if (!canBeDividedIntoTwo(groups, n, nextGroup, adj)) return false
    }
    return true
}

fun minimumTotalCost(nums1: IntArray, nums2: IntArray): Long {

    println(
        minimumTotalCost(
            intArrayOf(1, 2, 3, 4, 5), intArrayOf(1, 2, 3, 4, 5)
        )
    )

    return 0L
}

fun maxAbsValExpr(arr1: IntArray, arr2: IntArray): Int {

    println(
        maxAbsValExpr(
            intArrayOf(1, 2, 3, 4), intArrayOf(-1, 4, 5, 6)
        )
    )

    return 0
}

class Allocator(val n: Int) {

    val memory = IntArray(n) { -1 }
    val mIDMapping = hashMapOf<Int, MutableList<Int>>()

    fun allocate(size: Int, mID: Int): Int {
        var at = 0
        while (at < memory.size) {
            if (memory[at] == -1) {
                val st = at
                var c = 0
                while (at < memory.size && memory[at] == -1) {
                    c++
                    at++
                    if (c == size) break
                }
                if (c == size) {
                    if (mIDMapping[mID] == null) mIDMapping[mID] = mutableListOf()
                    for (i in st until st + size) {
                        memory[i] = mID
                        mIDMapping[mID]!!.add(i)
                    }
                }
                return st
            } else at++
        }
        return -1
    }

    fun free(mID: Int): Int {
        if (mIDMapping.containsKey(mID)) {
            val locs = mIDMapping[mID]!!
            for (l in locs) {
                memory[l] = -1
            }
            mIDMapping.remove(mID)
            return locs.size
        }
        return 0
    }

}


fun maxPoints(grid: Array<IntArray>, queries: IntArray): IntArray {

    val answer = IntArray(queries.size) { 0 }
    //value,at
    val queue = PriorityQueue(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            return o1[0] - o2[0]
        }
    })
    for (i in 0 until queries.size) {
        queue.add(intArrayOf(queries[i], i))
    }
    val visited = Array(grid.size) { BooleanArray(grid[0].size) { false } }
    var taken = 0
    val list = PriorityQueue(object : Comparator<SoFarValue> {
        override fun compare(o1: SoFarValue, o2: SoFarValue): Int {
            return o1.value - o2.value
        }
    })
    list.add(SoFarValue(grid[0][0], 0, 0))
    while (queue.isNotEmpty()) {
        val (value, at) = queue.poll()
        while (list.isNotEmpty()) {
            val peek = list.peek()
            if (peek.value < value) {
                taken++
                val poll = list.poll()
                visited[poll.i][poll.j] = true
                for (dir in dirs) {
                    val nI = dir[0] + peek.i
                    val nJ = dir[1] + peek.j
                    if (nI < 0 || nJ < 0 || nI == grid.size || nJ == grid[0].size || visited[nI][nJ]) continue
                    visited[nI][nJ] = true
                    list.add(SoFarValue(grid[nI][nJ], nI, nJ))
                }
            } else break
        }
        answer[at] = taken
    }
    return answer
}

data class SoFarValue(val value: Int, val i: Int, val j: Int)

fun takeSumOfPrimes(cur: Int): Int {
    var sum = 0
    var num = cur
    while (num != 1 && !isPrime(num)) {
        for (i in 2..num) {
            if (isPrime(i) && num % i == 0) {
                sum += i
                num /= i
            }
            if (num == 1 || i > num) break
        }
    }
    if (num != 1) sum += num
    return sum
}

fun cycleLengthQueries(n: Int, queries: Array<IntArray>): IntArray {

    val ans = IntArray(queries.size) { 0 }
    for (i in 0 until queries.size) {
        val (a, b) = queries[i]
        ans[i] = solve(a, b)
    }

    return ans
}

fun solve(a: Int, b: Int): Int {
    var edges = 0
    var i = a
    var j = b
    while (i != j) {
        if (i > j) {
            i /= 2
        } else j /= 2
        edges++
    }
    return ++edges
}

fun isPossible(n: Int, edges: List<List<Int>>): Boolean {

    val degrees = IntArray(n + 1) { 0 }
    val adj = Array(n + 1) { mutableListOf<Int>() }
    for ((to, from) in edges) {
        degrees[to]++
        degrees[from]++
        adj[to].add(from)
        adj[from].add(to)
    }
    var odd = 0
    val odds = mutableListOf<Int>()
    for (i in 1 until degrees.size) {
        if (degrees[i] % 2 != 0) {
            odd++
            odds.add(i)
        }
    }
    if (odd == 0) return true
    if (odd % 2 == 1 || odd >= 5) return false
    if (odd == 2) {
        if (!adj[odds[0]].contains(odds[1])) return true
        else {
            for (node in 1 until degrees.size) {
                val f = odds[0]
                val s = odds[1]
                if (!adj[node].contains(f) && !adj[node].contains(s)) return true
            }
        }
    }
    val ways = differentWays(odds[0], odds[1], odds[2], odds[3])
    for ((x, y, z, d) in ways) {
        if (!adj[x].contains(y) && !adj[z].contains(d)) return true
    }
    return false
}

fun differentWays(a: Int, b: Int, c: Int, d: Int): List<IntArray> {
    val list = mutableListOf<IntArray>()
    list.add(intArrayOf(a, b, c, d))
    list.add(intArrayOf(a, c, b, d))
    list.add(intArrayOf(a, d, b, c))
    return list
}

fun longestObstacleCourseAtEachPosition(obstacles: IntArray): IntArray {

    println(
        longestObstacleCourseAtEachPosition(
            intArrayOf(1, 2, 3, 2)
        )
    )

    return obstacles
}

fun minIncrements(n: Int, cost: IntArray): Int {
    println(
        minIncrements(
            7, intArrayOf(1, 5, 2, 2, 3, 3, 1)
        )
    )
    return 0
}


fun colorTheArray(n: Int, queries: Array<IntArray>): IntArray {
    val colors = IntArray(n) { 0 }
    val ans = IntArray(queries.size) { 0 }
    var same = 0
    for (i in 0 until queries.size) {
        val (index, color) = queries[i]
        if (colors[index] == color) {
            ans[i] = same
            continue
        }
        if (index - 1 >= 0 && colors[index] == colors[index - 1] && colors[index] != 0) same--
        if (index + 1 < colors.size && colors[index] == colors[index + 1] && colors[index] != 0) {
            same--
        }
        colors[index] = color
        if (index - 1 >= 0 && colors[index] == colors[index - 1]) same++
        if (index + 1 < colors.size && colors[index] == colors[index + 1]) {
            same++
        }
        ans[i] = same
    }
    return ans
}

class FrequencyTracker() {

    val freq = hashMapOf<Int, Int>()
    val values = hashMapOf<Int, Int>()

    fun add(number: Int) {
        val prev = freq.getOrDefault(number, 0)
        if (prev != 0) {
            values[prev] = values.getOrDefault(prev, 0) - 1
        }
        freq[number] = freq.getOrDefault(number, 0) + 1
        values[prev + 1] = values.getOrDefault(prev + 1, 0) + 1
    }

    fun deleteOne(number: Int) {
        val prev = freq.getOrDefault(number, 0)
        if (prev == 0) return
        freq[number] = prev - 1
        values[prev - 1] = values.getOrDefault(prev - 1, 0) + 1
        values[prev] = values.getOrDefault(prev, 0) - 1
    }

    fun hasFrequency(frequency: Int): Boolean {
        val fr = values.getOrDefault(frequency, 0)
        return fr != 0
    }

}


fun lca(a: Int, b: Int): Int {

    val visited = mutableSetOf<Int>()
    var len = 0
    var l = a
    visited.add(l)
    while (l != 1) {
        l /= 2
        visited.add(l)
    }
    var r = b
    while (!visited.contains(r)) {
        r /= 2
        len++
    }
    l = a
    while (l != r) {
        l /= 2
        len++
    }

    return ++len
}

fun minimizeSet(divisor1: Int, divisor2: Int, uniqueCnt1: Int, uniqueCnt2: Int): Int {

    println(
        minimizeSet(
            2, 7, 1, 3
        )
    )

    return 0
}

fun countAnagrams(s: String): Int {
    println(
        countAnagrams(
            "smuiquglfwdepzuyqtgujaisius ithsczpelfqp rjm"
        )
    )
    return 0
}

fun getAnagramCount(s: String): Long {
    var ans = 1L
    val freq = hashMapOf<Char, Int>()
    for (ch in s) {
        freq[ch] = freq.getOrDefault(ch, 0) + 1
    }
    var maxFreq = 0
    var char = 'a'
    for ((key, value) in freq) {
        if (value > maxFreq) {
            maxFreq = value
            char = key
        }
    }
    freq.remove(char)
    var cur = s.length
    while (cur > maxFreq) {
        ans *= cur
        cur--
    }


    return ans
}

fun captureForts(forts: IntArray): Int {
    var ans = 0
    for (i in 0 until forts.size) {
        if (forts[i] == 1) {
            var c = 0
            var l = i - 1
            while (l >= 0 && forts[l] == 0) {
                c++
                l--
            }
            var r = i + 1
            if (l >= 0 && forts[l] == -1) ans = Math.max(ans, c)
            c = 0
            while (r < forts.size && forts[r] == 0) {
                c++
                r++
            }
            if (r < forts.size && forts[r] == -1) ans = Math.max(ans, c)
        }
    }
    return ans
}

fun countPartitions(nums: IntArray, k: Int): Int {

    println(
        countPartitions(
            intArrayOf(1, 2, 3, 4), 4
        )
    )
    var total = 0L
    for (n in nums) {
        total += n
    }


    return 0
}

fun dpExplore(at: Int, nums: IntArray, total: Long, limit: Int): PartState {
    if (at == nums.size) return PartState(0, 0)
    //don't take
    val notTake = dpExplore(at + 1, nums, total, limit)
    //take
    val res = dpExplore(at + 1, nums, total, limit)
    var counter = 0L
    if (notTake.sum >= limit && total - notTake.sum >= limit) counter += 2
    val newSum = res.sum + nums[at]
    if (newSum >= limit && total - newSum >= limit) counter += 2
    return PartState(3, 0)
}

data class PartState(val sum: Int, val counter: Long)


fun minimumCost(start: IntArray, target: IntArray, specialRoads: Array<IntArray>): Int {

    val dist = hashMapOf<String, Int>()
    val queue = PriorityQueue(object : Comparator<DistState> {
        override fun compare(o1: DistState, o2: DistState): Int {
            return o1.dist - o2.dist
        }
    })
    queue.add(DistState(start[0], start[1], 0))
    val ans = Int.MAX_VALUE
    while (queue.isNotEmpty()) {
        val poll = queue.poll()
        if (poll.i == target[0] && poll.j == target[1]) return poll.dist
        var curDist = getDistanceRoad(intArrayOf(poll.i, poll.j), target) + poll.dist
        val dest = "${target[0]}|${target[1]}"
        if (curDist < dist.getOrDefault(dest, Int.MAX_VALUE)) {
            dist[dest] = curDist
            queue.add(DistState(target[0], target[1], curDist))
        }
        for ((x1, y1, x2, y2, cost) in specialRoads) {
            curDist = getDistanceRoad(intArrayOf(poll.i, poll.j), intArrayOf(x1, y1))
            val newCost = poll.dist + curDist + cost
            val newDest = "${x2}|${y2}"
            if (newCost < dist.getOrDefault(newDest, Int.MAX_VALUE)) {
                dist[newDest] = newCost
                queue.add(DistState(x2, y2, newCost))
            }
        }
        println(queue)
    }
    return ans
}

fun getDistanceRoad(start: IntArray, end: IntArray): Int {
    val abs1 = Math.abs(start[0] - end[0])
    val abs2 = Math.abs(start[1] - end[1])
    return abs1 + abs2
}

data class DistState(val i: Int, val j: Int, val dist: Int)

class MovieRentingSystem(val n: Int, val entries: Array<IntArray>) {

    val obj = MovieRentingSystem(
        3, arrayOf(
            intArrayOf(0, 1, 5),
            intArrayOf(0, 2, 6),
            intArrayOf(0, 3, 7),
            intArrayOf(1, 1, 4),
            intArrayOf(1, 2, 7),
            intArrayOf(2, 1, 5)
        )
    )
    //[shop,movie,price]

    //unrenting
    val unrenting = PriorityQueue(object : Comparator<Movie> {
        override fun compare(o1: Movie, o2: Movie): Int {
            if (o1.price == o2.price) {
                return o1.shop - o2.shop
            }
            return o1.price - o2.price
        }
    })

    //renting
    val renting = PriorityQueue(object : Comparator<Movie> {
        override fun compare(o1: Movie, o2: Movie): Int {
            if (o1.price == o2.price) if (o1.shop == o2.shop) return o1.movieId - o2.movieId
            else return o1.shop - o2.shop
            return o1.price - o2.price
        }
    })

    val priceMapping = hashMapOf<String, Int>()

    init {
        for ((shop, movie, price) in entries) {
            unrenting.add(Movie(shop, movie, price))
            val key = "$shop|$movie"
            priceMapping[key] = price
        }
    }


    fun search(movie: Int): List<Int> {
        val shops = mutableListOf<Int>()
        val pops = mutableListOf<Movie>()
        while (unrenting.isNotEmpty()) {
            val poll = unrenting.poll()
            if (poll.movieId == movie) {
                shops.add(poll.shop)
            }
            pops.add(poll)
            if (shops.size == 5) break
        }
        pops.forEach {
            unrenting.add(it)
        }
        return shops
    }

    fun rent(shop: Int, movie: Int) {
        val key = "$shop|$movie"
        val price = priceMapping[key]!!
        val obj = Movie(shop, movie, price)
        unrenting.remove(obj)
        renting.add(obj)
    }

    fun drop(shop: Int, movie: Int) {
        val key = "$shop|$movie"
        val price = priceMapping[key]!!
        val obj = Movie(shop, movie, price)
        unrenting.add(obj)
        renting.remove(obj)
    }

    fun report(): List<List<Int>> {
        val ans = mutableListOf<List<Int>>()
        val pops = mutableListOf<Movie>()
        while (renting.isNotEmpty()) {
            val pop = renting.poll()
            ans.add(listOf(pop.shop, pop.movieId))
            pops.add(pop)
            if (ans.size == 5) break
        }
        pops.forEach {
            renting.add(it)
        }
        return ans
    }

    data class Movie(val shop: Int, val movieId: Int, val price: Int)

}

fun minWastedSpace(packages: IntArray, boxes: Array<IntArray>): Int {

    val sortedBox = boxes.sortedWith(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            var l = 0
            var r = 0
            while (l < o1.size && r < o2.size && o1[l] == o2[r]) {
                l++
                r++
            }
            if (l == o1.size || r == o2.size) {
                return o1.size - o2.size
            }
            return o1[l] - o2[r]
        }
    })
    val sorted = packages.sorted().toIntArray()
    val prefix = LongArray(packages.size) { 0L }
    var sum = 0L
    for (i in 0 until prefix.size) {
        sum += sorted[i]
        prefix[i] = sum
    }
    var l = 0
    var r = sortedBox.size - 1
    var resWast = -1L
    while (l <= r) {
        val mid = (l + r) / 2
        val box = sortedBox[mid]
        val res = wasted(sorted, box, prefix)
        println("Box : ${box.toList()}, Res : $res, Mid : $mid")
        if (res == -1L) {
            l = mid + 1
        } else {
            resWast = res
            r = mid - 1
        }
    }
    if (resWast == -1L) return -1
    return resWast.toInt()
}

fun wasted(sorted: IntArray, box: IntArray, prefix: LongArray): Long {
    var wasted = 0L
    var l = 0
    for (b in box) {
        val maxMid = bs(sorted, b)
        if (maxMid == -1) continue
        if (maxMid < l) continue
        val cur = (b * (maxMid - l + 1)) - (prefix[maxMid] - if (l != 0) prefix[l - 1] else 0)
        wasted += cur
        wasted %= mod
        l = maxMid + 1
    }
    if (l < prefix.size) return -1L
    return wasted
}

fun getRange(prefix: LongArray, l: Int, r: Int): Long {
    if (l == 0) return prefix[r]
    return prefix[r] - prefix[l - 1]
}

fun bs(packages: IntArray, cur: Int): Int {
    var max = -1
    var l = 0
    var r = packages.size - 1
    while (l <= r) {
        val mid = l + r
        if (packages[mid] <= cur) {
            max = Math.max(max, mid)
            l = mid + 1
        } else {
            r = mid - 1
        }
    }
    return max
}

fun firstCompleteIndex(arr: IntArray, mat: Array<IntArray>): Int {
    val map = hashMapOf<Int, Pair<Int, Int>>()
    for (i in 0 until mat.size) {
        for (j in 0 until mat[i].size) {
            val value = mat[i][j]
            map[value] = Pair(i, j)
        }
    }
    val rows = mutableSetOf<Int>()
    val cols = mutableSetOf<Int>()
    for (i in 0 until arr.size) {
        val value = arr[i]
        val pair = map[value]!!
        rows.add(pair.first)
        cols.add(pair.second)
        if (rows.size == mat.size - 1) return i
        if (cols.size == mat[0].size - 1) return i
    }
    return 0
}

fun isWinner(player1: IntArray, player2: IntArray): Int {
    val sc1 = getScore(player1)
    val sc2 = getScore(player2)
    if (sc1 > sc2) return 1
    else if (sc1 < sc2) return 2
    return 0
}

fun getScore(nums: IntArray): Int {
    var score = nums[0]
    for (i in 1 until nums.size) {
        var double = false
        if (nums[i - 1] == 10) double = true
        if (i - 2 >= 0 && nums[i - 2] == 10) double = true
        score += if (double) nums[i] * 2 else nums[i]
    }
    return score
}

fun countOperationsToEmptyArray(nums: IntArray): Long {
    println(
        countOperationsToEmptyArray(
            intArrayOf(
                3, 4, -1
            )
        )
    )
    return 0L
}

fun findMaxFish(grid: Array<IntArray>): Int {
    var max = 0
    for (i in 0 until grid.size) {
        for (j in 0 until grid[i].size) {
            if (grid[i][j] != 0) max = Math.max(max, dfsFish(i, j, grid))
        }
    }
    return max
}

fun dfsFish(i: Int, j: Int, grid: Array<IntArray>): Int {
    if (i < 0 || j < 0 || i == grid.size || j == grid[i].size || grid[i][j] == 0) return 0
    var cur = grid[i][j]
    grid[i][j] = 0
    cur += dfsFish(i + 1, j, grid)
    cur += dfsFish(i, j + 1, grid)
    cur += dfsFish(i - 1, j, grid)
    cur += dfsFish(i, j - 1, grid)
    return cur
}

fun findThePrefixCommonArray(A: IntArray, B: IntArray): IntArray {
    val ans = IntArray(A.size) { 0 }
    var c = 0
    val freq = hashMapOf<Int, Int>()
    for (i in 0 until A.size) {
        val n1 = A[i]
        val n2 = B[i]
        freq[n1] = freq.getOrDefault(n1, 0) + 1
        freq[n2] = freq.getOrDefault(n2, 0) + 1
        if (n1 == n2) {
            if (freq[n1] == 2) c++
        } else {
            if (freq[n1] == 2) c++
            if (freq[n2] == 2) c++
        }
        ans[i] = c
    }
    return ans
}

fun maximizeSum(nums: IntArray, k: Int): Int {
    var max = nums.max()!!
    var sum = 0
    for (i in 0 until k) {
        sum += max
        max++
    }
    return sum
}

fun maximumRequests(n: Int, requests: Array<IntArray>): Int {
    backtracking(0, IntArray(n), 0, requests)
    return ansMax
}

var ansMax = 0
fun backtracking(at: Int, state: IntArray, counter: Int, requests: Array<IntArray>) {
    if (at == requests.size) {
        var candidate = true
        for (i in 0 until state.size) {
            if (state[i] != 0) {
                candidate = false
                break
            }
        }
        if (candidate) ansMax = Math.max(ansMax, counter)
        return
    }
    val (from, to) = requests[at]
    state[to]++
    state[from]--
    backtracking(at + 1, state, counter + 1, requests)
    state[to]--
    state[from]++
    backtracking(at + 1, state, counter, requests)
}

fun distanceLimitedPathsExist(n: Int, edgeList: Array<IntArray>, queries: Array<IntArray>): BooleanArray {

    println(
        distanceLimitedPathsExist(
            3, arrayOf(
                intArrayOf(0, 1, 2), intArrayOf(1, 2, 4), intArrayOf(2, 0, 8), intArrayOf(1, 0, 16)
            ), arrayOf(
                intArrayOf(0, 1, 2), intArrayOf(0, 2, 5)
            )
        )
    )


    return booleanArrayOf()
}

fun maxOutput(n: Int, edges: Array<IntArray>, price: IntArray): Long {
    val adj = Array(n) { mutableListOf<Int>() }
    for ((from, to) in edges) {
        adj[from].add(to)
        adj[to].add(from)
    }
    var ans = Long.MIN_VALUE
    val memo = hashMapOf<String, Long>()
    for (root in 0 until adj.size) {
        val maxPathSum = price[root] + dpExplore(root, -1, adj, price, memo)
        val dif = maxPathSum - price[root]
        ans = Math.max(ans, dif)
    }
    return ans
}

fun dpExplore(
    at: Int, parent: Int, adj: Array<MutableList<Int>>, prices: IntArray, memo: HashMap<String, Long>
): Long {
    var max = 0L
    val key = "$at|$parent"
    if (memo.containsKey(key)) return memo[key]!!
    for (n in adj[at]) {
        if (n != parent) {
            val res = dpExplore(n, at, adj, prices, memo)
            max = Math.max(max, res + prices[n])
        }
    }
    memo[key] = max
    return max
}

fun findMinimumTime(tasks: Array<IntArray>): Int {
    findMinimumTime(
        parseToArrayOfIntArray(
            "[[10,16,3],[10,20,5],[1,12,4],[8,11,2]]"
        )
    )

    var ans = 0
    val sorted = tasks.sortedWith(object : java.util.Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            if (o1[1] == o2[1]) return o2[0] - o1[0]
            return o1[1] - o2[1]
        }
    })
    sorted.forEach {
        println(it.toList())
    }
    for (i in 0 until tasks.size) {
        val cur = sorted[i]
        val end = cur[1]
        val duration = cur[2]
        if (duration == 0) continue
        ans += duration
        for (j in i + 1 until tasks.size) {
            val simu = sorted[j]
            if (simu[0] > end) continue
            val dura = simu[2]
            val allowed = end - simu[0] + 1
            val min = Math.min(dura, Math.min(allowed, duration))
            simu[2] -= min
        }
    }
    println("Ans : $ans")
    return ans

}

fun minOperations(nums: IntArray): Int {
    println(
        minOperations(
            intArrayOf(2, 6, 3, 4)
        )
    )
    var ones = 0
    for (i in 0 until nums.size) {
        if (nums[i] == 1) ones++
    }
    if (ones != 0) {
        return nums.size - ones
    }

    return 0
}


fun nthSuperUglyNumber(n: Int, primes: IntArray): Int {
    if (n == 1) return 1
    val queue = PriorityQueue(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            val m1 = o1[0] * o1[1]
            val m2 = o2[0] * o2[1]
            return m1 - m2
        }
    })
    for (i in 0 until primes.size) {
        queue.add(intArrayOf(1, primes[i]))
    }
    var cur = n
    var prev = cur
    while (cur > 0) {
        val poll = queue.poll()
        println(poll[1] * primes[poll[0]])
        if (poll[1] * primes[poll[0]] == prev) continue
        cur--
        prev = poll[1] * primes[poll[0]]
        if (poll[0] + 1 < primes.size) queue.add(intArrayOf(poll[0] + 1, poll[1]))
    }
    return prev
}

fun maximumTastiness(price: IntArray, k: Int): Int {
    price.sort()
    var l = 0
    var r = Int.MAX_VALUE
    var ans = 0
    while (l <= r) {
        val mid = l + (r - l) / 2
        if (numPairs(price, mid, k)) {
            ans = Math.max(ans, mid)
            l = mid + 1
        } else r = mid - 1
    }
    return ans
}

fun numPairs(nums: IntArray, dif: Int, k: Int): Boolean {
    var c = 0
    var min = (dif * -1)
    for (i in 0 until nums.size) {
        if (nums[i] - min >= dif) {
            min = nums[i]
            c++
            println("Min : $min")
        }
    }
    println("C : $c")
    return c >= k
}

fun dpExplore(at: Int, gr1: Int, gr2: Int, nums: IntArray, k: Int, memo: HashMap<String, Long>): Long {
    if (at == nums.size) {
        if (gr1 >= k && gr2 >= k) return 1L
        return 0L
    }
    val key = "$at|$gr1|$gr2"
    if (memo.containsKey(key)) return memo[key]!!
    val add1 = dpExplore(at + 1, gr1 + nums[at], gr2, nums, k, memo) % mod
    val add2 = dpExplore(at + 1, gr1, gr2 + nums[at], nums, k, memo) % mod
    val res = add1 + add2
    memo[key] = res % mod
    return res % mod
}

fun isItPossible(word1: String, word2: String): Boolean {
    if (word1.toSet().size == word2.toSet().size) return true
    val freq1 = hashMapOf<Char, Int>()
    val freq2 = hashMapOf<Char, Int>()
    for (w in word1) freq1[w] = freq1.getOrDefault(w, 0) + 1
    for (w in word2) freq2[w] = freq2.getOrDefault(w, 0) + 1

    for (a in freq1.keys.toMutableList()) {
        for (b in freq2.keys.toMutableList()) {
            swap(a, b, freq1, freq2)
            if (freq1.size == freq2.size) return true
            swap(b, a, freq1, freq2)
        }
    }
    return false
}

fun swap(a: Char, b: Char, freq1: HashMap<Char, Int>, freq2: HashMap<Char, Int>) {
    freq1[a] = freq1.getOrDefault(a, 0) - 1
    freq2[b] = freq2.getOrDefault(b, 0) - 1
    if (freq1[a] == 0) freq1.remove(a)
    if (freq2[b] == 0) freq2.remove(b)
    freq1[b] = freq1.getOrDefault(b, 0) + 1
    freq2[a] = freq2.getOrDefault(a, 0) + 1
}


fun getCommon(nums1: IntArray, nums2: IntArray): Int {
    nums1.sort()
    nums2.sort()
    var at1 = 0
    var at2 = 0
    while (at1 < nums1.size && at2 < nums2.size) {
        if (nums1[at1] == nums2[at2]) return nums1[at1]
        if (nums1[at1] < nums2[at2]) at1++
        else at2++
    }
    return -1
}

fun minOperations(nums1: IntArray, nums2: IntArray, k: Int): Long {
    val ans = mutableListOf<Int>()
    var sum = 0L
    for (i in 0 until nums1.size) {
        if (nums2[i] != nums1[i]) {
            val abs = Math.abs(nums1[i] - nums2[i])
            ans.add((nums1[i] - nums2[i]) / k)
            sum += ans[ans.size - 1]
            if (abs % k != 0) return -1
        }
    }
    if (sum != 0L) return -1
    val sorted = ans.sorted().toMutableList()
    var l = 0
    var r = sorted.size - 1
    while (l < r) {
        val min = Math.min(Math.abs(sorted[l]), sorted[r])
        sum += min
        sorted[l] += min
        sorted[r] -= min
        if (sorted[l] == 0) l++
        if (sorted[r] == 0) r--
    }
    return sum
}

fun maxScore(nums1: IntArray, nums2: IntArray, k: Int): Long {
    println(
        maxScore(
            intArrayOf(4, 2, 3, 1, 1), intArrayOf(7, 5, 10, 9, 6), 1
        )
    )
    var at = 0
    var max = 0
    for (i in 0 until nums1.size) {
        val score = nums1[i] * nums2[i]
        if (score > max) {
            max = score
            at = i
        }
    }
    var sum = nums1[at].toLong()
    var min = nums2[at]
    //  sum/min
    val queue = PriorityQueue(object : java.util.Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            val sc1 = (sum + o1[0]) * Math.min(min, o1[1])
            val sc2 = (sum + o2[0]) * Math.min(min, o2[1])
            return if (sc1 <= sc2) 1
            else -1
        }
    })
    for (i in 0 until nums2.size) {
        if (i != at) {
            queue.add(intArrayOf(nums1[i], nums2[i]))
        }
    }
    var cur = k - 1
    while (cur > 0) {
        val poll = queue.poll()
        sum += poll[0]
        min = Math.min(poll[1], min)
        cur--
    }
    return sum * min
}

fun isReachable(targetX: Int, targetY: Int): Boolean {

    return true
}

fun monkeyMove(n: Int): Int {
    val power = Math.pow(2.toDouble(), n.toDouble())
    return power.toInt()
}

fun putMarbles(weights: IntArray, k: Int): Long {


    return 0L
}


fun getSubarrayBeauty(nums: IntArray, k: Int, x: Int): IntArray {
    val ans = mutableListOf<Int>()
    val freq = hashMapOf<Int, Int>()
    var l = 0
    for (r in 0 until nums.size) {
        val n = nums[r]
        freq[n] = freq.getOrDefault(n, 0) + 1
        if (r >= k - 1) {
            var c = 0
            var small = 0
            for (i in -50..-1) {
                c += freq[i] ?: 0
                if (c >= x) {
                    small = i
                    break
                }
            }
            ans.add(small)
            freq[nums[l]] = freq.getOrDefault(nums[l], 0) - 1
            l++
        }
    }
    println(ans.toList())
    return ans.toIntArray()
}


fun findDelayedArrivalTime(arrivalTime: Int, delayedTime: Int): Int {
    return (arrivalTime + delayedTime) % 24
}

fun dpMax(at: Int, bags: Int, weights: IntArray): Long {
    if (at == weights.size) {
        return if (bags == 0) 0 else Long.MIN_VALUE
    }
    if (bags < 0) return Long.MIN_VALUE
    var min = Long.MIN_VALUE
    for (i in at + 1 until weights.size) {
        val res = dpMax(i + 1, bags - 1, weights)
        if (res != Long.MIN_VALUE) {
            val score = weights[i] + weights[at]

        }
    }
    return 0
}

fun countQuadruplets(nums: IntArray): Long {
    println(
        countQuadruplets(
            intArrayOf(1, 3, 2, 4, 5)
        )
    )
    return 0L
}

fun minCost(basket1: IntArray, basket2: IntArray): Long {
    println(
        minCost(
            intArrayOf(4, 4, 4, 4, 3), intArrayOf(5, 5, 5, 5, 3)
        )
    )
    val s1 = mutableListOf<Int>()
    val s2 = mutableListOf<Int>()
    val freq = hashMapOf<Int, Int>()
    for (i in 0 until basket1.size) {
        val b1 = basket1[i]
        val b2 = basket2[i]
        freq[b1] = freq.getOrDefault(b1, 0) + 1
        freq[b2] = freq.getOrDefault(b2, 0) + 1
    }
    for ((key, value) in freq) {
        if (value % 2 == 1) return -1
    }
    basket1.sort()
    basket2.sort()
    var at1 = 0
    var at2 = 0
    while (at1 < basket1.size) {
        if (at2 < basket2.size && basket1[at1] == basket2[at2]) {
            at1++
            at2++
        } else if (at2 < basket2.size) {
            if (basket1[at1] > basket2[at2]) {
                s2.add(basket2[at2++])
            } else {
                s1.add(basket1[at1++])
            }
        } else {
            s1.add(basket1[at1])
            at1++
        }
    }
    while (at2 < basket2.size) s2.add(basket2[at2++])
    var ans = 0L

    return ans
}

fun minImpossibleOR(nums: IntArray): Int {
    println(
        minImpossibleOR(
            intArrayOf(5, 3, 2)
        )
    )
    return 0
}

fun minimizeSum(nums: IntArray): Int {
    if (nums.size <= 3) return 0
    nums.sort()
    println(nums.toList())
    val removeBoth = Math.abs(Math.abs(nums[1]) - Math.abs(nums[nums.size - 2]))
    val removeLeft = Math.abs(Math.abs(nums[1]) - Math.abs(nums[nums.size - 1]))
    val removeRight = Math.abs(Math.abs(nums[0]) - Math.abs(nums[nums.size - 2]))
    val twoLeft = Math.abs(Math.abs(nums[2]) - Math.abs(nums[nums.size - 1]))
    val twoRight = Math.abs(Math.abs(nums[0]) - Math.abs(nums[nums.size - 3]))
    val res = listOf(removeBoth, removeLeft, removeRight, twoLeft, twoRight)
    println("Res : $res")
    return res.min()!!
}

fun handleQuery(nums1: IntArray, nums2: IntArray, queries: Array<IntArray>): LongArray {

    println(
        handleQuery(
            intArrayOf(1, 0, 1), intArrayOf(0, 0, 0), arrayOf(
                intArrayOf(1, 1, 1), intArrayOf(2, 1, 0), intArrayOf(3, 0, 0)
            )
        )
    )
    return longArrayOf()
}

fun squareFreeSubsets(nums: IntArray): Int {

    val primes = intArrayOf(2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    val bitLoc = hashMapOf<Int, Int>()
    for (i in 0 until primes.size) {
        bitLoc[primes[i]] = i
    }
    val divPrimes = mutableListOf<List<Int>>()
    for (i in 0 until nums.size) {
        var cur = nums[i]
        val curPrimes = mutableListOf<Int>()
        if (cur == 1) curPrimes.add(1)
        var at = 0
        while (cur != 1) {
            if (cur % primes[at] == 0) {
                curPrimes.add(primes[at])
                cur /= primes[at]
            } else at++
        }
        if (curPrimes.size == curPrimes.toSet().size) {
            divPrimes.add(curPrimes)
        }
    }
    val ans = dpExplore(0, 0, divPrimes, bitLoc)
    return (ans % mod).toInt()
}

fun dpExplore(at: Int, bit: Int, divPrimes: MutableList<List<Int>>, bitLoc: HashMap<Int, Int>): Long {
    if (at == divPrimes.size) return 0
    var ans = 0L
    val curPrimes = divPrimes[at]
    var nextBit = bit
    var canTake = true
    if (curPrimes[0] != 1) {
        for (prime in curPrimes) {
            val loc = bitLoc[prime]!!
            if (nextBit and (1 shl loc) == 0) {
                nextBit = nextBit or (1 shl loc)
            } else {
                canTake = false
                break
            }
        }
    } else nextBit = 1
    if (canTake) {
        ans += dpExplore(at + 1, nextBit, divPrimes, bitLoc)
    }
    ans += dpExplore(at + 1, bit, divPrimes, bitLoc)
    ans %= mod
    return ans
}

fun findTheString(lcp: Array<IntArray>): String {

    println(
        findTheString(
            arrayOf(
                intArrayOf(4, 3, 2, 1), intArrayOf(3, 3, 2, 1), intArrayOf(2, 2, 2, 1), intArrayOf(1, 1, 1, 1)
            )
        )
    )

    return ""
}

fun maxNumOfMarkedIndices(nums: IntArray): Int {
    val sorted = nums.sorted()
    var end = sorted.size / 2
    var ans = 0
    println(sorted)
    for (start in 0 until nums.size / 2) {
        val cur = sorted[start]
        while (end < nums.size && cur * 2 > sorted[end]) {
            end++
        }
        if (end < nums.size) {
            end++
            ans += 2
        } else break
    }
    return ans
}

fun minimumTime(tasks: Array<IntArray>): Int {
    println(
        minimumTime(
            arrayOf(
                intArrayOf(15, 20, 1), intArrayOf(3, 18, 2), intArrayOf(4, 12, 6)
            )
        )
    )
    var ans = 0
    val sorted = tasks.sortedWith(object : java.util.Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            if (o1[0] == o2[0]) return o1[1] - o2[1]
            return o1[0] - o2[0]
        }
    })
    sorted.forEach {
        println(it.toList())
    }
    for (i in 0 until tasks.size) {
        val cur = sorted[i]
        val end = cur[1]
        val duration = cur[2]
        println("Duration : $duration")
        if (duration == 0) continue
        ans += duration
        for (j in i + 1 until tasks.size) {
            val simu = sorted[j]
            if (simu[0] > end) break
            val dura = simu[2]
            val allowed = end - simu[0] + 1
            val min = Math.min(dura, Math.min(allowed, duration))
            simu[2] -= min
        }
    }
    return ans
}

fun minimumVisitedCells(grid: Array<IntArray>): Int {

    val rows = Array(grid.size) { mutableListOf<Int>() }
    val cols = Array(grid[0].size) { mutableListOf<Int>() }
    for (i in 0 until rows.size) {
        rows[i] = (0 until grid[0].size).toMutableList()
    }
    for (i in 0 until cols.size) {
        cols[i] = (0 until grid.size).toMutableList()
    }
    val queue = LinkedList<IntArray>()
    queue.add(intArrayOf(0, 0))
    var steps = 1
    while (queue.isNotEmpty()) {
        val size = queue.size
        for (i in 0 until size) {
            val p = queue.poll()
            if (p[0] == grid.size - 1 && p[1] == grid[0].size - 1) return steps
            val value = grid[p[0]][p[1]]
            val maxCol = value + p[1]
            val colsList = rows[p[0]].toMutableList()
            for (col in colsList) {
                if (col > maxCol) break
                queue.add(intArrayOf(p[0], col))
                rows[p[0]].remove(col)
            }
            val rowsList = cols[p[1]].toMutableList()
            val maxRow = value + p[0]
            for (row in rowsList) {
                if (row > maxRow) break
                queue.add(intArrayOf(row, p[1]))
                cols[p[1]].remove(row)
            }
        }
        steps++
    }

    return -1
}

fun rootCount(edges: Array<IntArray>, guesses: Array<IntArray>, k: Int): Int {
    println(
        rootCount(
            arrayOf(
                intArrayOf(0, 1), intArrayOf(1, 2), intArrayOf(1, 3), intArrayOf(4, 2)
            ), arrayOf(
                intArrayOf(1, 3), intArrayOf(0, 1), intArrayOf(1, 0), intArrayOf(2, 4)
            ), 3
        )
    )
    val adj = Array(edges.size + 1) { mutableListOf<Int>() }
    for ((from, to) in edges) {
        adj[from].add(to)
        adj[to].add(from)
    }
    val guessAdj = Array(edges.size + 1) { mutableSetOf<Int>() }
    for ((parent, children) in guesses) {
        guessAdj[parent].add(children)
    }
    var ans = 0
    val memo = IntArray(edges.size + 1) { -1 }
    for (root in 0..edges.size) {
        val res = countCorrectGuesses(root, -1, adj, guessAdj, memo)
        if (res >= k) ans++
    }

    return ans
}

fun countCorrectGuesses(
    at: Int, parent: Int, adj: Array<MutableList<Int>>, guessAdj: Array<MutableSet<Int>>, memo: IntArray
): Int {
    if (memo[at] != -1) return memo[at]
    var counter = 0
    for (n in adj[at]) {
        if (n != parent) {
            if (guessAdj[at].contains(n)) counter++
            counter += countCorrectGuesses(n, at, adj, guessAdj, memo)
        }
    }
    memo[at] = counter
    return counter
}

fun countGroups(sorted: Array<IntArray>): Int {
    val treeMap = TreeMap<Int, IntArray>()
    for ((from, to) in sorted) {
        val floor = treeMap.floorKey(from)
        if (floor != null) {
            val prevEnding = treeMap[floor]!!
            if (prevEnding[1] >= from) {
                treeMap[floor] = intArrayOf(prevEnding[0], Math.max(prevEnding[1], to))
            } else {
                treeMap[from] = intArrayOf(from, to)
            }
        } else {
            treeMap[from] = intArrayOf(from, to)
        }
    }
    return treeMap.size
}


fun collectTheCoins(coins: IntArray, edges: Array<IntArray>): Int {
    if (coins.size <= 3) return 0
    val adj = Array(coins.size) { mutableListOf<Int>() }
    for ((from, to) in edges) {
        adj[from].add(to)
        adj[to].add(from)
    }
    val queue = LinkedList<Int>()
    for (i in 0 until adj.size) {
        if (adj[i].size == 1 && coins[i] == 0) {
            queue.add(i)
        }
    }
    while (queue.isNotEmpty()) {
        val leaf = queue.poll()
        val parent = adj[leaf][0]
        adj[leaf].remove(parent)
        adj[parent].remove(leaf)
        if (adj[parent].size == 1 && coins[parent] == 0) {
            queue.add(parent)
        }
    }
    for (i in 0 until adj.size) {
        if (adj[i].size == 1) {
            queue.add(i)
        }
    }
    while (queue.isNotEmpty()) {
        val leaf = queue.poll()
        val parent = adj[leaf][0]
        adj[parent].remove(leaf)
        adj[leaf].remove(parent)
        if (adj[parent].size == 1) {
            val grandParent = adj[parent][0]
            adj[parent].remove(grandParent)
            adj[grandParent].remove(parent)
        }
    }
    var count = 0
    for (i in 0 until adj.size) {
        if (adj[i].size != 0) {
            count += subCount(i, -1, adj)
            break
        }
    }
    if (count == 0) return 0
    return (count - 1) * 2
}

fun subCount(i: Int, parent: Int, adj: Array<MutableList<Int>>): Int {
    var c = 1
    for (n in adj[i]) {
        if (n != parent) c += subCount(n, i, adj)
    }
    return c
}

fun nthUglyNumber(n: Int, a: Int, b: Int, c: Int): Int {
    var l = 0
    var r = Int.MAX_VALUE
    var ans = Int.MAX_VALUE
    while (l <= r) {
        val mid = l + (r - l) / 2
        if (countDivisions(mid, a.toLong(), b.toLong(), c.toLong()) >= n) {
            ans = Math.min(ans, mid)
            r = mid - 1
        } else l = mid + 1
    }
    return ans
}

fun countDivisions(mid: Int, a: Long, b: Long, c: Long): Long {
    val divA = mid / a
    val divB = mid / b
    val divC = mid / c
    val lcmAB = lcm(a, b)
    val lcmBC = lcm(b, c)
    val lcmAC = lcm(a, c)
    val lcmABC = lcm(lcm(a, b), c)
    var count = divA + divB + divC - (mid / lcmAB) - (mid / lcmBC) - (mid / lcmAC) + (mid / lcmABC)
    return count.toLong()
}

fun lcm(a: Long, b: Long): Long {
    return (a * b * 1L) / findGCD(a, b)
}

fun findGCD(a: Long, b: Long): Long {
    if (b == 0L) return a
    var small = a
    var bigger = b
    while (small != 0L) {
        val temp = bigger
        bigger = small
        small = temp % small
    }
    return bigger
}


fun minKBitFlips(nums: IntArray, k: Int): Int {

    var cur = 0
    val map = hashMapOf<Int, Int>()
    var ans = 0
    for (i in 0 until nums.size) {
        cur += map[i] ?: 0
        val bit = nums[i]
        if (bit == 1) {
            if (cur % 2 == 1) {
                ans++
                cur++
                val next = i + k
                map[next] = map.getOrDefault(next, 0) - 1
            }
        } else {
            if (cur % 2 == 0) {
                ans++
                cur++
                val next = i + k
                map[next] = map.getOrDefault(next, 0) - 1
            }
        }
    }

    return ans
}

fun findValidSplit(nums: IntArray): Int {

    val max = nums.max()
    val primes = mutableListOf<Int>()
    for (i in 2..max) {
        if (isPrime(i)) primes.add(i)
    }
    val primesFreq = Array(nums.size) { hashMapOf<Int, Int>() }
    for (i in 0 until nums.size) {
        val freq = getPrimesEach(nums[i], primes)
        primesFreq[i] = freq
    }
    val allFreq = hashMapOf<Int, Int>()
    for (i in 0 until nums.size) {
        val map = primesFreq[i]
        for ((key, value) in map) {
            allFreq[key] = allFreq.getOrDefault(key, 0) + value
        }
    }
    val prefix = hashMapOf<Int, Int>()
    for (i in 0 until nums.size - 1) {
        val map = primesFreq[i]
        for ((key, value) in map) {
            prefix[key] = prefix.getOrDefault(key, 0) + value
            allFreq[key] = allFreq.getOrDefault(key, 0) - value
            if (allFreq[key] == 0) allFreq.remove(key)
        }
        var hasFound = false
        for ((key, value) in prefix) {
            if (allFreq.containsKey(key)) {
                hasFound = true
                break
            }
        }
        if (!hasFound) return i
    }
    return -1
}

val cachedPrimes = hashMapOf<Int, HashMap<Int, Int>>()

fun getPrimesEach(num: Int, primes: MutableList<Int>): HashMap<Int, Int> {
    if (cachedPrimes.containsKey(num)) return cachedPrimes[num]!!
    var cur = num
    var at = 0
    val freq = hashMapOf<Int, Int>()
    while (cur != 1) {
        if (cur % primes[at] == 0) {
            val n = primes[at]
            freq[n] = freq.getOrDefault(num, 0) + 1
            cur /= primes[at]
        } else at++
    }
    cachedPrimes[num] = freq
    return freq
}

fun dfs(at: Int, parent: Int, trimed: Array<MutableList<Int>>) {
    println("Trim : $at")
    for (n in trimed[at]) {
        if (n != parent) {
            dfs(n, at, trimed)
        }
    }
}

fun trimTree(
    at: Int, parent: Int, adj: Array<MutableList<Int>>, coins: IntArray, trimed: Array<MutableList<Int>>
): Boolean {
    var hasLeaf = false
    if (coins[at] == 1) hasLeaf = true
    for (n in adj[at]) {
        if (n != parent) {
            val res = trimTree(n, at, adj, coins, trimed)
            if (res) {
                trimed[n].add(at)
                trimed[at].add(n)
            }
            hasLeaf = hasLeaf or res
        }
    }
    return hasLeaf
}

fun minReverseOperations(n: Int, p: Int, banned: IntArray, k: Int): IntArray {

    val bannedLoc = BooleanArray(n) { false }
    for (b in banned) {
        bannedLoc[b] = true
    }
    val ans = IntArray(n) { Int.MAX_VALUE }
    ans[p] = 0
    val visited = BooleanArray(n) { false }
    visited[p] = true
    val queue = LinkedList<Int>()
    queue.add(p)
    while (queue.isNotEmpty()) {
        val size = queue.size


    }
    return banned
}

fun reverseWithSlidingWindow(at: Int, n: Int, window: Int) {
    println(
        minReverseOperations(
            4, 2, intArrayOf(0, 1, 3), 1
        )
    )


}


fun minimumTotalPrice(n: Int, edges: Array<IntArray>, price: IntArray, trips: Array<IntArray>): Int {

    println(
        minimumTotalPrice(
            4, arrayOf(
                intArrayOf(0, 1), intArrayOf(1, 2), intArrayOf(1, 3)
            ), intArrayOf(2, 2, 10, 6), arrayOf(
                intArrayOf(0, 3), intArrayOf(2, 1), intArrayOf(2, 3)
            )
        )
    )
    val adj = Array(n) { mutableListOf<Int>() }
    for ((from, to) in edges) {
        adj[from].add(to)
        adj[to].add(from)
    }
    val trimed = Array(n) { mutableSetOf<Int>() }
    val visited = Array(n) { false }
    for ((start, end) in trips) {
        shouldAddEdge(start, -1, end, adj, trimed, visited)
    }
    var ans = 0
    for (node in 0 until visited.size) {
        if (visited[node]) {
            var curMin = dpExplore(node, false, -1, price, trimed, visited)
            curMin = Math.min(curMin, dpExplore(node, true, -1, price, trimed, visited))
            ans += curMin
        }
    }
    return ans
}

fun dpExplore(
    at: Int, parentHalf: Boolean, parent: Int, scores: IntArray, trimed: Array<MutableSet<Int>>, visited: Array<Boolean>
): Int {
    visited[at] = false
    var fullScore = Int.MAX_VALUE
    var halfScore = Int.MAX_VALUE
    if (parentHalf) {
        fullScore = scores[at]
        for (n in trimed[at]) {
            if (n != parent) {
                fullScore += dpExplore(n, false, at, scores, trimed, visited)
            }
        }
    } else {
        halfScore = scores[at] / 2
        for (n in trimed[at]) {
            if (n != parent) {
                halfScore += dpExplore(n, true, at, scores, trimed, visited)
            }
        }
    }

    return Math.min(halfScore, fullScore)
}

fun dfs(at: Int, parent: Int, trimed: Array<MutableSet<Int>>) {
    for (n in trimed[at]) {
        if (n != parent) dfs(n, at, trimed)
    }
}

fun shouldAddEdge(
    at: Int, parent: Int, to: Int, adj: Array<MutableList<Int>>, trimed: Array<MutableSet<Int>>, visited: Array<Boolean>
): Boolean {
    if (at == to) return true
    for (n in adj[at]) {
        if (n != parent && shouldAddEdge(n, at, to, adj, trimed, visited)) {
            visited[at] = true
            visited[n] = true
            trimed[at].add(n)
            trimed[n].add(at)
            return true
        }
    }
    return false
}

fun addMinimum(word: String): Int {
    return dpExplore(3, 0, "b")
}

fun dpExplore(prev: Int, at: Int, word: String): Int {
    if (at == word.length) {
        val last = word[word.length - 1] - 'a' + 1
        if (last == 1) return 2
        else if (last == 2) return 1
        else return 0
    }
    val cur = word[at] - 'a' + 1
    var next = prev + 1
    if (next == 4) next = 1
    if (cur == next) return dpExplore(next, at + 1, word)
    return 1 + dpExplore(next, at, word)
}

fun maxDivScore(nums: IntArray, divisors: IntArray): Int {
    var ans = 0
    var maxDiv = 0
    val set = divisors.toSet()
    for (s in set) {
        var c = 0
        for (n in nums) {
            if (n % s == 0) c++
        }
        if (c > maxDiv) {
            maxDiv = c
            ans = c
        } else if (c == maxDiv) {
            ans = Math.min(ans, s)
        }
    }
    return ans
}

fun minimizeMax(nums: IntArray, p: Int): Int {
    nums.sort()
    var l = 0
    var r = Int.MAX_VALUE
    while (l < r) {
        val mid = l + (r - l) / 2
        if (isPossiblePairs(nums, mid, p)) {
            r = mid
        } else l = mid + 1
    }
    return l
}

fun isPossiblePairs(nums: IntArray, mid: Int, p: Int): Boolean {
    var at = 0
    var pairs = 0
    while (at < nums.size) {
        if (at + 1 < nums.size && nums[at + 1] - nums[at] <= mid) {
            pairs++
            at += 2
        } else at++
    }
    return pairs >= p
}


fun findPrefixScore(nums: IntArray): LongArray {
    val ans = LongArray(nums.size) { 0 }
    var max = nums[0]
    var maxSoFar = LongArray(nums.size) { 0 }
    for (i in 0 until nums.size) {
        max = Math.max(max, nums[i])
        maxSoFar[i] = nums[i] + max.toLong()
    }
    var sum = 0L
    for (i in 0 until ans.size) {
        sum += maxSoFar[i]
        ans[i] = sum
    }
    return ans
}

fun findColumnWidth(grid: Array<IntArray>): IntArray {
    val cols = mutableListOf<Int>()
    for (col in 0 until grid[0].size) {
        var max = 1
        for (row in 0 until grid.size) {
            val cur = "${grid[col][row]}"
            max = Math.max(max, cur.length)
        }
        cols.add(max)
    }
    return cols.toIntArray()
}

fun getPermutation(n: Int, k: Int): String {
    val nums = IntArray(n)
    for (i in 0 until nums.size) {
        nums[i] = i + 1
    }
    permutation(0, nums)
    list.sorted()
    return list[k - 1]
}

val list = mutableListOf<String>()

fun permutation(at: Int, arr: IntArray) {
    if (at == arr.size) {
        list.add(arr.joinToString(""))
    }
    for (i in at until arr.size) {
        swapNum(at, i, arr)
        permutation(at + 1, arr)
        swapNum(at, i, arr)
    }
}

fun swapNum(i: Int, j: Int, arr: IntArray) {
    val temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp
}

fun grayCode(n: Int): List<Int> {
    var cur = 0
    for (i in 0 until n) cur = cur or (1 shl i)
    val visited = BooleanArray(cur + 1) { false }
    dfs(visited, mutableListOf(), 0, 0, n)
    return gray
}

var gray = mutableListOf<Int>()

fun dfs(visited: BooleanArray, curList: MutableList<Int>, at: Int, counter: Int, n: Int) {
    if (gray.isNotEmpty()) return
    if (counter == visited.size) {
        if (isOneBitDifference(curList[0], curList[curList.size - 1])) {
            gray = curList.toMutableList()
        }
        return
    }
    for (i in 0 until n) {
        //set
        val set = at or (1 shl i)
        if (isOneBitDifference(set, at) && !visited[set]) {
            visited[set] = true
            curList.add(set)
            dfs(visited, curList, set, counter + 1, n)
            curList.removeAt(curList.size - 1)
            visited[set] = false
        }
        //unset
        val unset = at xor (1 shl i)
        if (isOneBitDifference(unset, at) && !visited[unset]) {
            visited[unset] = true
            curList.add(unset)
            dfs(visited, curList, unset, counter + 1, n)
            curList.removeAt(curList.size - 1)
            visited[unset] = false
        }
    }
}


fun isOneBitDifference(cur: Int, next: Int): Boolean {
    val xor = cur xor next
    return xor != 0 && xor and (xor - 1) == 0
}

fun numTrees(n: Int): Int {
    return dfsTrees(1, n, n + 1)
}

fun dfsTrees(l: Int, r: Int, max: Int): Int {
    if (l == r) return 0
    if (l <= 0 || r == max) return 0
    var c = 1
    for (i in l..r) {
        c += dfsTrees(l, i - 1, max) + dfsTrees(i + 1, r, max)
    }
    return c
}

fun canCompleteCircuit(gas: IntArray, cost: IntArray): Int {
    var s1 = 0L
    var s2 = 0L
    for (g in gas) s1 += g
    for (c in cost) s2 += c
    if (s2 < s1) return -1
    var at = 0
    var ans = -1
    while (at < gas.size) {
        var sum = gas[at]
        ans = at
        if (sum - cost[at] < 0) {
            at++
            continue
        }
        while (at < gas.size && sum - cost[at] >= 0) {
            sum -= cost[at]
            at++
            if (at < cost.size) sum += gas[at]
        }
    }
    return ans
}

class LRUCache(val capacity: Int) {

    var lastNode: Node? = null
    var newNode: Node? = null
    val map = hashMapOf<Int, Node>()

    fun get(key: Int): Int {
        if (!map.containsKey(key)) return -1
        val oldNode = map[key]!!
        if (oldNode == lastNode) {
            if (lastNode?.next != null) lastNode = lastNode?.next
        }
        val prev = oldNode.prev
        val next = oldNode.next
        oldNode.next = null
        next?.prev = prev
        prev?.next = next
        newNode?.next = oldNode
        oldNode.prev = newNode
        newNode = oldNode
        oldNode.next = null
        return oldNode.value
    }

    fun put(key: Int, value: Int) {
        if (lastNode == null) {
            val node = Node(value, key)
            lastNode = node
            newNode = node
            map[key] = node
        } else {
            if (map.containsKey(key)) {
                val oldNode = map[key]!!
                if (oldNode == lastNode) {
                    lastNode = lastNode!!.next
                }
                val prev = oldNode.prev
                val next = oldNode.next
                oldNode.next = null
                next?.prev = prev
                oldNode.prev = newNode
                newNode = oldNode
                oldNode.next = null
                oldNode.value = value
            } else {
                if (map.size == capacity) {
                    val removeKey = lastNode!!.key
                    lastNode = lastNode!!.next
                    lastNode?.prev = null
                    map.remove(removeKey)
                }
                val node = Node(value, key)
                map[key] = node
                node.prev = newNode
                newNode?.next = node
                newNode = newNode?.next
            }
        }
    }

    data class Node(var value: Int, var key: Int) {
        var prev: Node? = null
        var next: Node? = null
    }

    fun goThrough() {
        var cur = lastNode
        while (cur != null) {
            println("Value : ${cur.value}")
            cur = cur.next
        }
        println("Finishing")
    }

}

fun findMin(nums: IntArray): Int {
    return findDFS(0, nums.size - 1, nums)
}

fun findDFS(l: Int, r: Int, nums: IntArray): Int {
    if (l == r) return nums[l]
    println("Left : $l : Right : $r")
    var left = l
    var right = r
    while (left < right) {
        val mid = (left + right) / 2
        if (mid + 1 < nums.size && nums[mid] > nums[mid + 1]) {
            return nums[mid + 1]
        } else if (nums[mid] == nums[mid + 1]) {
            val lMin = findDFS(left, mid, nums)
            val rMin = findDFS(mid + 1, right, nums)
            return Math.min(lMin, rMin)
        } else if (nums[mid] < nums[nums.size - 1]) {
            right = mid
        } else left = mid + 1
    }
    return nums[0]
}

fun fractionToDecimal(numerator: Int, denominator: Int): String {
    println(
        fractionToDecimal(
            4, 333
        )
    )
    return ""
}

fun containsNearbyAlmostDuplicate(nums: IntArray, indexDiff: Int, valueDiff: Int): Boolean {
    val treeMap = TreeMap<Int, Int>()
    for (i in 0 until indexDiff) {
        val cur = nums[i]
        val floor = treeMap.floorKey(cur)
        if (floor != null) {
            val diff = cur - floor
            if (diff <= valueDiff) return true
        }
        val ceil = treeMap.ceilingKey(cur)
        if (ceil != null) {
            val diff = ceil - cur
            if (diff <= valueDiff) return true
        }
        treeMap[cur] = treeMap.getOrDefault(cur, 0) + 1
    }
    var l = 0
    for (r in indexDiff until nums.size) {
        var cur = nums[r]
        val floor = treeMap.floorKey(cur)
        if (floor != null) {
            val diff = cur - floor
            if (diff <= valueDiff) return true
        }
        val ceil = treeMap.ceilingKey(cur)
        if (ceil != null) {
            val diff = ceil - cur
            if (diff <= valueDiff) return true
        }
        treeMap[cur] = treeMap.getOrDefault(cur, 0) + 1
        cur = nums[l++]
        treeMap[cur] = treeMap.getOrDefault(cur, 0) - 1
        if (treeMap[cur] == 0) treeMap.remove(cur)
    }
    return false
}

fun countDigitOne(n: Int): Int {
    var ans = if (n >= 1) 1 else 0
    for (i in 1..9) ans += dfsCounting(if (i == 1) 1 else 0, i.toLong(), n.toLong())
    return ans
}

fun dfsCounting(ones: Int, cur: Long, max: Long): Int {
    if (cur > max) return 0
    var ans = 0
    for (i in 0..9) {
        val next = cur * 10L + i
        if (next <= max) {
            ans += ones
            if (i == 1) ans++
            ans += dfsCounting(ones + if (i == 1) 1 else 0, next, max)
        }
    }
    return ans
}

fun addOperators(num: String, target: Int): List<String> {
    val ans = mutableSetOf<String>()
    val builder = StringBuilder()
    for (i in 0 until num.length - 1) {
        builder.append(num[i])
        dfs(i + 1, builder.toString(), num, target, ans)
    }
    return ans.toList()
}

fun dfs(at: Int, cur: String, str: String, target: Int, ans: MutableSet<String>) {
    if (at == str.length) {
        if (evaluate(cur) == target.toLong()) {
            ans.add(cur)
        }
        return
    }
    val operators = arrayOf('+', '-', '*')
    val builder = StringBuilder()
    for (i in at until str.length) {
        builder.append(str[i])
        if (builder[0] == '0' && builder.length > 1) break
        operators.forEach {
            dfs(i + 1, cur + "$it" + "$builder", str, target, ans)
        }
    }
}

fun evaluate(str: String): Long {
    val stack = Stack<String>()
    var at = 0
    val builder = StringBuilder()
    while (at < str.length && str[at].isDigit()) builder.append(str[at++])
    stack.add(builder.toString())
    while (at < str.length) {
        if (str[at] == '*') {
            at++
            val pop = stack.pop().toLong()
            builder.clear()
            while (at < str.length && str[at].isDigit()) {
                builder.append(str[at++])
            }
            stack.push((pop * builder.toString().toLong()).toString())
        } else {
            stack.push(str[at].toString())
            at++
            builder.clear()
            while (at < str.length && str[at].isDigit()) {
                builder.append(str[at++])
            }
            stack.push(builder.toString())
        }
    }
    var res = stack[0].toLong()
    at = 1
    while (at < stack.size) {
        val op = stack[at]
        val num = stack[at + 1]
        if (op == "+") {
            res += num.toLong()
        } else {
            res -= num.toLong()
        }
        at += 2
    }
    return res
}

fun dpExplore(at: Int, pairs: Int, nums: IntArray): Int {

    return 0
}

fun dpExplore(i: Int, j: Int, grid: Array<IntArray>, memo: Array<IntArray>): Int {
    if (i == grid.size - 1 && j == grid[0].size - 1) return 1
    if (memo[i][j] != -1) return memo[i][j]
    var min = Int.MAX_VALUE
    val col = Math.min(grid[i][j] + j, grid[0].size - 1)
    for (c in j + 1..col) {
        val res = dpExplore(i, c, grid, memo)
        if (res != Int.MAX_VALUE) min = Math.min(min, res + 1)
    }
    val row = Math.min(grid.size - 1, grid[i][j] + i)
    for (r in i + 1..row) {
        val res = dpExplore(r, j, grid, memo)
        if (res != Int.MAX_VALUE) min = Math.min(min, res + 1)
    }
    memo[i][j] = min
    return min
}

fun distance(nums: IntArray): LongArray {
    val map = hashMapOf<Int, MutableList<Int>>()
    for (i in 0 until nums.size) {
        val cur = nums[i]
        if (map[cur] == null) map[cur] = mutableListOf()
        map[cur]!!.add(i)
    }
    val answer = LongArray(nums.size) { 0L }
    for ((key, value) in map) {
        solve(answer, value.toIntArray())
    }
    return answer
}

fun solve(answer: LongArray, indices: IntArray) {
    if (indices.size == 1) return
    val preLeft = LongArray(indices.size)
    val preRight = LongArray(indices.size)
    var sum = 0L
    for (i in 0 until indices.size) {
        sum += indices[i]
        preLeft[i] = sum
    }
    sum = 0L
    for (i in indices.size - 1 downTo 0) {
        sum += indices[i]
        preRight[i] = sum
    }
    for (i in 0 until indices.size) {
        val cur = indices[i]
        var ans = 0L
        if (i - 1 >= 0) {
            val left = preLeft[i - 1]
            ans += Math.abs((i) * cur.toLong() - left)
        }
        if (i + 1 < indices.size) {
            val right = preRight[i + 1]
            ans += Math.abs((indices.size - i - 1) * cur.toLong() - right)
        }
        answer[cur] = ans
    }
}

fun diagonalPrime(nums: Array<IntArray>): Int {
    var max = 0
    for (i in 0 until nums.size) {
        if (isPrime(nums[i][i])) {
            max = Math.max(max, nums[i][i])
        }
        if (isPrime(nums[i][nums.size - i - 1])) {
            max = Math.max(max, nums[i][nums.size - i - 1])
        }
    }
    return max
}

fun maxCoins(nums: IntArray): Int {
    println(
        maxCoins(
            intArrayOf(3, 1, 5, 8)
        )
    )
    return 0
}


fun maxNumber(nums1: IntArray, nums2: IntArray, k: Int): IntArray {

    println(
        maxNumber(
            intArrayOf(3, 4, 6, 5), intArrayOf(9, 1, 2, 5, 8, 3), 5
        )
    )

    return nums1
}

fun wiggleSort(nums: IntArray) {
    val sorted = nums.sorted()
    var at = 0
    var mid = if (nums.size % 2 == 0) nums.size / 2 else nums.size / 2 + 1
    for (i in 0 until nums.size) {
        if (i % 2 == 0) {
            nums[i] = sorted[at++]
        } else {
            nums[i] = sorted[mid++]
        }
    }
}

fun countRangeSum(nums: IntArray, lower: Int, upper: Int): Int {
    println(
        countRangeSum(
            intArrayOf(
                -2, 5, -1
            ), -2, 2
        )
    )
    return 0
}

fun minPatches(nums: IntArray, n: Int): Int {
    println(
        minPatches(
            intArrayOf(1, 5, 10), 20
        )
    )



    return 0
}

fun isSelfCrossing(distance: IntArray): Boolean {

    println(
        isSelfCrossing(
            intArrayOf(
                2, 1, 1, 2
            )
        )
    )

    return false
}

fun canMeasureWater(j1: Int, j2: Int, t: Int): Boolean {
    if (t > j1 && t > j2) return false

    return false
}

fun dfs(j1: Int, j2: Int, t: Int): Boolean {
    if (j1 == t || j2 == t) return true

    return false
}

fun superPow(a: Int, b: IntArray): Int {
    val mod = 1337

    return a
}

fun deserialize(s: String): NestedInteger {
    println(s)
    if (s[0] == '[') {
        val obj = NestedInteger()
        val without = s.substring(1, s.length - 1)
        val stack = Stack<Char>()
        val builder = StringBuilder()
        for (char in without) {
            if (char == ',') {
                if (stack.isEmpty()) {
                    val res = deserialize(builder.toString())
                    obj.add(res)
                    builder.clear()
                } else {
                    builder.append(char)
                }
            } else if (char == '[') {
                builder.append(char)
                stack.push('[')
            } else if (char == ']') {
                builder.append(char)
                stack.pop()
            } else builder.append(char)
        }
        if (builder.isNotEmpty()) {
            val res = deserialize(builder.toString())
            obj.add(res)
        }
        return obj
    }
    return NestedInteger(s.toInt())
}

class NestedInteger {
    // Constructor initializes an empty nested list.
    constructor()

    // Constructor initializes a single integer.
    constructor(value: Int)

    // @return true if this NestedInteger holds a single integer, rather than a nested list.
    fun isInteger(): Boolean {
        return false
    }

    // @return the single integer that this NestedInteger holds, if it holds a single integer
    // Return null if this NestedInteger holds a nested list
    fun getInteger(): Int? {
        return null
    }

    // Set this NestedInteger to hold a single integer.
    fun setInteger(value: Int): Unit {}

    // Set this NestedInteger to hold a nested list and adds a nested integer to it.
    fun add(ni: NestedInteger): Unit {}

    // @return the nested list that this NestedInteger holds, if it holds a nested list
    // Return null if this NestedInteger holds a single integer
    fun getList(): List<NestedInteger>? {
        return emptyList()
    }
}

fun lastRemaining(n: Int): Int {
    val arr = IntArray(n / 2) { 0 }
    var at = 0
    for (i in 2..n step 2) {
        arr[at++] = i
    }
    return recursion(arr, false)[0]
}

fun recursion(arr: IntArray, fromStart: Boolean): IntArray {
    if (arr.size == 1) return arr
    val newArr = IntArray(arr.size / 2) { 0 }
    if (fromStart) {
        var at = 0
        for (i in 1 until arr.size step 2) {
            newArr[at++] = arr[i]
        }
        return recursion(newArr, false)
    } else {
        var at = newArr.size - 1
        for (i in arr.size - 2 downTo 0 step 2) {
            newArr[at--] = arr[i]
        }
        return recursion(newArr, true)
    }
}

fun isRectangleCover(rectangles: Array<IntArray>): Boolean {
    println(
        isRectangleCover(
            arrayOf(
                intArrayOf(1, 1, 3, 3),
                intArrayOf(3, 1, 4, 2),
                intArrayOf(3, 2, 4, 4),
                intArrayOf(1, 3, 2, 4),
                intArrayOf(2, 3, 3, 4)
            )
        )
    )


    return false
}


fun isIdealPermutation(nums: IntArray): Boolean {

    val obj = DTreeWrapper()
    obj.update(nums[nums.size - 1])
    var global = 0L
    for (i in nums.size - 2 downTo 0) {
        val cur = obj.getRangeValue(nums[i] - 1)
        global += cur
        obj.update(nums[i])
    }
    var local = 0L
    for (i in 0 until nums.size - 1) {
        if (nums[i] > nums[i + 1]) local++
    }

    return local == global
}

fun maxRotateFunction(nums: IntArray): Int {
    if (nums.size == 1) return 0
    val prefRight = IntArray(nums.size) { 0 }
    var sum = 0
    for (i in nums.size - 1 downTo 0) {
        sum += nums[i]
        prefRight[i] = sum
    }
    val scoresRight = IntArray(nums.size) { 0 }
    var score = 0
    for (i in nums.size - 1 downTo 0) {
        score += (nums[i] * i)
        scoresRight[i] = score
    }
    val prefLeft = IntArray(nums.size)
    val scoreLeft = IntArray(nums.size)
    score = 0
    sum = 0
    for (i in 0 until prefRight.size) {
        sum += nums[i]
        score += (nums[i] * i)
        prefLeft[i] = sum
        scoreLeft[i] = score
    }
    var max = scoresRight[0] - prefRight[0]
    for (i in 1 until nums.size) {
        val cur1 = scoresRight[i] - prefRight[i] * i
        val multi = (nums.size - i)
        val cur2 = scoreLeft[i - 1] + (prefLeft[i - 1] * multi)
        max = Math.max(max, cur1 + cur2)
    }
    return max
}

fun toHex(num: Int): String {
    if (num == 0) return "0"
    var cur = num
    if (num < 0) {
        val binary = Integer.toBinaryString(num * (-1))
        println(binary)
        val front = "0".repeat(32 - binary.length)
        val fullBit = front + binary
        println(fullBit.length)
        val builder = StringBuilder(fullBit)
        for (i in 0 until builder.length) {
            builder.setCharAt(i, if (builder[i] == '1') '0' else '1')
        }
    }
    return convertPositiveToHex(cur)
}

val hex = arrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f')

fun convertPositiveToHex(num: Int): String {
    var cur = num
    val builder = StringBuilder()
    while (cur != 0) {
        val char = hex[cur % 16]
        builder.append(char)
        cur = cur shr 4
    }
    return builder.reversed().toString()
}

fun findNthDigit(n: Int): Int {
    var max = 0
    var l = 0L
    var r = Int.MAX_VALUE.toLong()
    while (l <= r) {
        val mid = l + (r - l) / 2L
        val res = countDigits(mid)
        if (res <= n) {
            max = Math.max(max, mid.toInt())
            l = mid + 1
        } else r = mid - 1
    }
    var maxUsed = countDigits(max + 2L)
    var cur = max + 2L
    var ans = -1
    while (maxUsed >= n) {
        val rev = "$cur".reversed()
        for (i in 0 until rev.length) {
            if (maxUsed == n.toLong()) {
                ans = rev[i] - '0'
                break
            }
            maxUsed--
        }
        if (ans != -1) break
        cur--
    }
    return ans
}

fun countDigits(max: Long): Long {
    var counter = 0L
    val maxLen = "$max".length
    for (i in 1 until maxLen) {
        val digits = "9" + "0".repeat(i - 1)
        counter += digits.toInt() * i
    }
    val remain = max - ("1" + "0".repeat(maxLen - 1)).toInt() + 1
    return counter + (remain * (maxLen))
}


fun sumOfDistancesInTree(n: Int, edges: Array<IntArray>): IntArray {

    return intArrayOf()
}

fun minSumOfLengths(arr: IntArray, target: Int): Int {
    val pre = hashMapOf<Int, Int>()
    var sum = 0
    for (i in 0 until arr.size) {
        sum += arr[i]
        pre[sum] = i
    }
    var ans = Int.MAX_VALUE
    sum = 0
    for (i in 0 until arr.size) {
        sum += arr[i]
        if (pre.containsKey(sum - target)) {

        }
    }
    return 0
}


fun originalDigits(s: String): String {
    println(
        originalDigits(
            "fviefour"
        )
    )
    val freq = hashMapOf<Char, Int>()
    for (ch in s) {
        freq[ch] = freq.getOrDefault(ch, 0) + 1
    }
    //one three five seven nine
    //zero
    val nums = IntArray(10) { 0 }
    if ((freq['z'] ?: 0) > 0) removeUniqueChars(0, 'z', "zero", freq, nums)
    //six
    if ((freq['x'] ?: 0) > 0) removeUniqueChars(6, 'x', "six", freq, nums)
    //two
    if ((freq['w'] ?: 0) > 0) removeUniqueChars(2, 'w', "two", freq, nums)
    //four
    if ((freq['u'] ?: 0) > 0) removeUniqueChars(4, 'u', "four", freq, nums)
    println("After four : $freq")
    //eight
    if ((freq['g'] ?: 0) > 0) removeUniqueChars(8, 'g', "eight", freq, nums)
    //one
    if ((freq['o'] ?: 0) > 0) removeUniqueChars(1, 'o', "one", freq, nums)
    //five
    if ((freq['v'] ?: 0) > 0) removeUniqueChars(5, 'v', "five", freq, nums)
    if ((freq['h'] ?: 0) > 0) removeUniqueChars(3, 'h', "three", freq, nums)
    if ((freq['s'] ?: 0) > 0) removeUniqueChars(7, 's', "seven", freq, nums)
    if ((freq['i'] ?: 0) > 0) removeUniqueChars(9, 'i', "nine", freq, nums)
    val ans = java.lang.StringBuilder()
    for (i in 0 until nums.size) {
        ans.append("$i".repeat(nums[i]))
    }
    return ans.toString()
}

fun removeUniqueChars(num: Int, char: Char, engish: String, freq: HashMap<Char, Int>, nums: IntArray) {
    nums[num] = freq[char]!!
    for (ch in engish) {
        freq[ch] = freq.getOrDefault(ch, 0) - nums[num]
        if (freq[ch] == 0) freq.remove(ch)
    }
}

fun eraseOverlapIntervals(intervals: Array<IntArray>): Int {
    println(
        eraseOverlapIntervals(
            arrayOf(
                intArrayOf(1, 2), intArrayOf(2, 3), intArrayOf(3, 4), intArrayOf(1, 3)
            )
        )
    )
    return 0
}

fun findGoodStrings(n: Int, s1: String, s2: String, evil: String): Int {


    return 0
}

class SolutionFlipping(val m: Int, val n: Int) {

    var max = m * n
    val swap = hashMapOf<Int, Int>()

    fun flip(): IntArray {
        var rand = Random.nextInt(max--)
        rand = swap.getOrDefault(rand, max)
        swap[rand] = max
        return intArrayOf(rand / n, rand % n)
    }

    fun reset() {
        max = m * n
    }

}

fun findMEX(arr: IntArray): Long {
    val pre = LongArray(arr.size) { 0 }
    var sum = 0L
    for (i in 0 until arr.size) {
        sum += arr[i]
        pre[i] = sum
    }
    val t = (arr.size * (arr.size + 1L)) / 2L
    var c = 0
    for (i in 0 until arr.size) {
        for (j in i until arr.size) {
            val rS = getRange(i, j, pre)
            val dist = (j - i + 1)
            val tS = (dist * (dist + 1L)) / 2L
            if (rS == tS) c++
        }
    }
    return t - c
}

fun getRange(l: Int, r: Int, prefix: LongArray): Long {
    if (l == 0) return prefix[r]
    return prefix[r] - prefix[l - 1]
}

fun dpExplore(lastEnding: Int, at: Int, sorted: Array<IntArray>, memo: HashMap<String, Int>): Int {
    if (at == sorted.size) return 0
    val key = "$lastEnding|$at"
    if (memo.containsKey(key)) return memo[key]!!
    var min = 1 + dpExplore(lastEnding, at + 1, sorted, memo)
    val curInter = sorted[at]
    if (lastEnding <= curInter[0]) {
        val res = dpExplore(curInter[1], at + 1, sorted, memo)
        min = Math.min(min, res)
    }
    memo[key] = min
    return min
}

fun findKthNumber(n: Int, k: Int): Int {
    var c = 0
    for (i in 1..9) {
        val stack = Stack<Int>()
        stack.push(i)
        while (stack.isNotEmpty()) {
            val pop = stack.pop()
            c++
            if (c == k) return pop
            for (j in 9 downTo 0) {
                val next = pop * 10 + j
                if (next <= n) stack.push(next)
            }
        }
    }
    return 0
}

fun numberOfBoomerangs(points: Array<IntArray>): Int {
    var ans = 0
    for (i in 0 until points.size) {
        val map = hashMapOf<Double, Int>()
        for (j in 0 until points.size) {
            if (i != j) {
                val distance = getDistance(points[i], points[j])
                map[distance] = map.getOrDefault(distance, 0) + 1
            }
        }
        for ((key, value) in map) {
            if (value >= 2) {
                val difPoints = (value * (value - 1)) / 2
                ans += (difPoints * 2)
            }
        }
    }
    return ans
}

fun getDistance(a: IntArray, b: IntArray): Double {
    val difX = Math.abs(a[0] - b[0])
    val difY = Math.abs(a[1] - b[1])
    val powX = Math.pow(difX.toDouble(), 2.0)
    val powY = Math.pow(difY.toDouble(), 2.0)
    return Math.sqrt(powX + powY)
}

fun fourSumCount(nums1: IntArray, nums2: IntArray, nums3: IntArray, nums4: IntArray): Int {
    val freq1 = hashMapOf<Int, Int>()
    val freq2 = hashMapOf<Int, Int>()
    for (i in 0 until nums1.size) {
        for (j in 0 until nums2.size) {
            val sum = (nums1[i] + nums2[j])
            freq1[sum] = freq1.getOrDefault(sum, 0) + 1
        }
    }
    for (i in 0 until nums3.size) {
        for (j in 0 until nums4.size) {
            val sum = -nums3[i] - nums4[j]
            freq2[sum] = freq2.getOrDefault(sum, 0) + 1
        }
    }
    var ans = 0
    for ((key, value) in freq1) {
        if (freq2[key] != null) {
            ans += (value * freq2[key]!!)
        }
    }
    return ans
}

fun circularArrayLoop(nums: IntArray): Boolean {
    println(
        circularArrayLoop(
            intArrayOf(
                18, 21, 12
            )
        )
    )
    val component = IntArray(nums.size) { -1 }
    var comp = 0
    for (i in 0 until nums.size) {
        if (component[i] == -1) {
            if (containLoop(i, nums, comp, component)) {
                println("Starting : $i")
                return true
            }
        }
        comp++
    }

    return false
}

fun containLoop(at: Int, nums: IntArray, group: Int, component: IntArray): Boolean {
    if (component[at] != -1) {
        if (component[at] == group) return true
        return false
    }
    println("At : $at")
    component[at] = group
    if (Math.abs(nums[at]) % nums.size == 0) return false
    if (nums[at] > 0) {
        val next = (at + mod) % nums.size
        return containLoop(next, nums, group, component)
    }
    val abs = Math.abs(mod)
    if (at - abs >= 0) {
        val back = at - abs
        return containLoop(back, nums, group, component)
    }
    val back = nums.size + (at - abs)
    return containLoop(back, nums, group, component)
}

fun getMaxRepetitions(s1: String, n1: Int, s2: String, n2: Int): Int {

    println(
        getMaxRepetitions(
            "acb", 4, "ab", 2
        )
    )


    return 0
}

fun validIPAddress(queryIP: String): String {

    println(
        validIPAddress(
            "001:0db8:85a3:0:0:8A2E:0370:7334"
        )
    )
    return ""
}

class SolutionRand : SolBase() {

    fun rand10(): Int {
        var idx: Int
        do {
            val row = rand7()
            val col = rand7()
            idx = (row - 1) * col
        } while (idx > 40)
        return 1 + (idx % 10)
    }

}

open class SolBase() {

    fun rand7(): Int {
        return Random.nextInt(1, 8)
    }

}

class SolutionCircle(val radius: Double, val x_center: Double, val y_center: Double) {


    fun randPoint(): DoubleArray {
        var x = Random.nextDouble(radius * 2 + 0.00001)
        var y = Random.nextDouble(radius * 2 + 0.00001)
        while (!isInCircle(x, y, x_center)) {
            x = Random.nextDouble(radius * 2 + 0.00001)
            y = Random.nextDouble(radius * 2 + 0.00001)
        }
        return doubleArrayOf(x + x_center, y_center)
    }

    private fun isInCircle(x: Double, y: Double, radius: Double): Boolean {
        val absX = Math.abs(x - x_center)
        val absY = Math.abs(y - y_center)
        val powX = Math.pow(absX, 2.0)
        val powY = Math.pow(absY, 2.0)
        return powX + powY <= Math.pow(radius, 2.0)
    }

}

fun largestPalindrome(n: Int): Int {
    println(
        largestPalindrome(
            2
        )
    )

    return 0
}

fun PredictTheWinner(nums: IntArray): Boolean {
    println(
        PredictTheWinner(
            intArrayOf(
                1, 5, 233, 7
            )
        )
    )
    return false
}

fun findCriticalAndPseudoCriticalEdges(n: Int, edges: Array<IntArray>): List<List<Int>> {

    println(
        findCriticalAndPseudoCriticalEdges(
            5, arrayOf(
                intArrayOf(0, 1, 1),
                intArrayOf(1, 2, 1),
                intArrayOf(2, 3, 2),
                intArrayOf(0, 3, 2),
                intArrayOf(0, 4, 3),
                intArrayOf(3, 4, 3),
                intArrayOf(1, 4, 6)
            )
        )
    )
    return emptyList()
}

fun reversePairs(nums: IntArray): Int {
    val obj = DynamicTree(Int.MAX_VALUE * 4L + 2)
    var ans = 0
    for (i in nums.size - 1 downTo 0) {
        val upper = nums[i].toLong() + Int.MAX_VALUE
        val res = obj.getRangeQuery(upper + 1, Int.MAX_VALUE * 4L + 2)
        ans += res
        obj.updateQueryValue(((nums[i].toLong() * 2) + Int.MAX_VALUE))
    }
    return ans
}

data class SNode(val node: Long, val l: Long, val r: Long) {
    var left: SNode? = null
    var right: SNode? = null
    var freq = 0

    fun extend(at: Long, l: Long, r: Long) {
        if (left == null) {
            val mid = (l + r) / 2
            left = SNode(at * 2 + 1, l, mid)
            right = SNode(at * 2 + 2, mid + 1, r)
        }
    }

}

class DynamicTree(val upper: Long) {

    var root = SNode(0, 0, upper + 1)

    fun updateQueryValue(value: Long) {
        updateHelper(0, 0, upper + 1, root, value)
    }

    fun updateHelper(at: Long, l: Long, r: Long, root: SNode, value: Long) {
        if (l == r && value == l) {
            root.freq++
            return
        }
        if (r < value || value < l) return
        root.freq++
        val mid = (l + r + 0L) / 2L
        root.extend(at, l, r)
        updateHelper(at * 2 + 1, l, mid, root.left!!, value)
        updateHelper(at * 2 + 2, mid + 1, r, root.right!!, value)
    }

    fun getRangeQuery(lower: Long, upper: Long): Int {
        return rangeHelperQueryFreq(0L, 0, this.upper, lower, upper, root)
    }

    fun rangeHelperQueryFreq(at: Long, l: Long, r: Long, lower: Long, upper: Long, root: SNode?): Int {
        if (root == null) return 0
        if (r < lower || upper < l) return 0
        if (lower <= l && r <= upper) {
            return root.freq
        }
        val mid = (l + r + 0L) / 2
        val leftFreq = rangeHelperQueryFreq(at * 2 + 1, l, mid, lower, upper, root.left)
        val rightFreq = rangeHelperQueryFreq(at * 2 + 2, mid + 1, r, lower, upper, root.right)
        return leftFreq + rightFreq
    }

    fun dfs() {
        dfsHelper(root)
    }

    private fun dfsHelper(root: SNode?) {
        if (root == null) return
        println("At : ${root.node} and Freq : ${root.freq}")
        dfsHelper(root.left)
        dfsHelper(root.right)
    }

}

fun miceAndCheese(reward1: IntArray, reward2: IntArray, k: Int): Int {

    val freq1 = hashMapOf<Int, Int>()
    val locations = hashMapOf<Int, MutableList<Int>>()
    val freq2 = hashMapOf<Int, Int>()
    for (i in 0 until reward1.size) {
        val r = reward1[i]
        freq1[r] = freq1.getOrDefault(r, 0) + 1
        if (locations[r] == null) {
            locations[r] = mutableListOf()
        }
        locations[r]!!.add(i)
    }
    for (r in reward2) {
        freq2[r] = freq2.getOrDefault(r, 0) + 1
    }
    val types = mutableListOf<IntArray>()
    for ((type, freq) in freq1) {
        val score = type * freq
        types.add(intArrayOf(score, type))
    }
    val sorted = types.sortedWith(object : java.util.Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            return o2[0] - o1[0]
        }
    })
    var ans = 0
    val visited = Array(reward2.size) { false }
    for (i in 0 until k) {
        val (score, type) = sorted[i]
        ans += score
        for (l in locations[type]!!) {
            visited[l] = true
        }
    }
    freq1.clear()
    for (i in 0 until reward2.size) {
        if (visited[i]) continue
        freq1[reward2[i]] = freq1.getOrDefault(reward2[i], 0) + 1
    }
    types.clear()
    for ((type, freq) in freq1) {
        val score = type * freq
        types.add(intArrayOf(score, type))
    }
    val sorted2 = types.sortedWith(object : java.util.Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            return o2[0] - o1[0]
        }
    })
    for (i in 0 until Math.min(k, sorted2.size)) {
        val (score, type) = sorted2[i]
        ans += score
    }
    return ans
}


fun findMatrix(nums: IntArray): List<List<Int>> {
    val answer = mutableListOf<List<Int>>()
    val freq = hashMapOf<Int, Int>()
    for (n in nums) {
        freq[n] = freq.getOrDefault(n, 0) + 1
    }
    while (true) {
        val row = mutableListOf<Int>()
        val keys = freq.keys.toList()
        for (k in keys) {
            row.add(k)
            freq[k] = freq.getOrDefault(k, 0) - 1
            if (freq[k] == 0) freq.remove(k)
        }
        if (row.isEmpty()) break
        answer.add(row)
    }
    return answer
}

fun findTheLongestBalancedSubstring(s: String): Int {
    var max = 0
    for (i in 0 until s.length) {
        var at = i
        var c = 0
        while (at < s.length && s[at] == '0') {
            c++
            at++
        }
        var ones = 0
        while (at < s.length && s[at] == '1') {
            ones++
            at++
        }
        max = Math.max(max, Math.min(c, ones) * 2)
    }
    return max
}


class ST(val max: Long) {

    val board = IntArray(getSTSize())

    private fun getSTSize(): Int {
        var power = 1
        while (power < max) {
            power *= 2
        }
        return 2 * power - 1
    }

}

class SolutionRandom(val rects: Array<IntArray>) {

    fun pick(): IntArray {
        return intArrayOf()
    }

}

class SolutionFlip(val m: Int, val n: Int) {

    val flipped = mutableSetOf<Location>()
    fun flip(): IntArray {
        var chosen = Random.nextInt(0, m * n)
        var curObj = Location(chosen / n, chosen % n)
        while (flipped.contains(curObj)) {
            chosen = Random.nextInt(0, m * n)
            curObj = Location(chosen / n, chosen % n)
        }
        flipped.add(curObj)
        return intArrayOf(curObj.i, curObj.j)
    }

    fun reset() {
        flipped.clear()
    }

    data class Location(val i: Int, val j: Int)

}

fun makeSubKSumEqual(arr: IntArray, k: Int): Long {
    println(
        makeSubKSumEqual(
            intArrayOf(
                1, 4, 1, 3
            ), 2
        )
    )
    val sorted = arr.sorted()
    var median = -1
    if (arr.size % 2 == 1) {
        median = arr[arr.size / 2]
    } else {
        val left = arr[arr.size / 2 - 1]
        val right = arr[arr.size / 2]
        median = (left + right) / 2
    }
    val targetSum = median * k
    println("TargetSum : $targetSum")

    return 0L
}

fun findShortestCycle(n: Int, edges: Array<IntArray>): Int {
    println(
        findShortestCycle(
            7, arrayOf(
                intArrayOf(0, 1),
                intArrayOf(1, 2),
                intArrayOf(2, 0),
                intArrayOf(3, 4),
                intArrayOf(4, 5),
                intArrayOf(5, 6),
                intArrayOf(6, 3)
            )
        )
    )
    val adj = Array(n) { mutableListOf<Int>() }
    for ((a, b) in edges) {
        adj[a].add(b)
        adj[b].add(a)
    }
    val visited = Array(n) { false }
    for (i in 0 until visited.size) {
        if (!visited[i]) {
            backtracking(i, -1, adj, 0, hashMapOf(), visited)
        }
    }
    if (minLen == Int.MAX_VALUE) return -1
    return minLen
}

val seen = mutableSetOf<String>()
var minLen = Int.MAX_VALUE

fun backtracking(
    at: Int, parent: Int, adj: Array<MutableList<Int>>, len: Int, map: HashMap<Int, Int>, visited: Array<Boolean>
) {
    if (map.containsKey(at)) {
        val prev = map[at]!!
        val dist = len - prev
        minLen = Math.min(minLen, dist)
        return
    }
    val key = "$at|$len"
    if (seen.contains(key)) return
    seen.add(key)
    visited[at] = true
    map[at] = len
    for (n in adj[at]) {
        if (n != parent) {
            backtracking(n, at, adj, len + 1, map, visited)
        }
    }
    map.remove(at)
}

fun maximumCostSubstring(s: String, chars: String, vals: IntArray): Int {
    var max = Int.MIN_VALUE
    var curScore = 0
    val map = hashMapOf<Char, Int>()
    for (i in 0 until chars.length) {
        val char = chars[i]
        val score = vals[i]
        map[char] = score
    }
    for (char in s) {
        val curValue = map[char] ?: (char - 'a' + 1)
        curScore += curValue
        max = Math.max(max, curScore)
        curScore = Math.max(0, curScore)
    }
    return max
}

fun findRotateSteps(ring: String, key: String): Int {
    println(
        findRotateSteps(
            "godding", "godding"
        )
    )
    return 0
}

fun dpExplore(atKey: Int, key: String, atRing: Int, ring: String, memo: Array<IntArray>): Int {
    if (atKey == key.length) return 0
    if (memo[atKey][atRing] != -1) return memo[atKey][atRing]
    val target = key[atKey]
    if (ring[atRing] == target) return dpExplore(atKey + 1, key, atRing, ring, memo)
    var min = Int.MAX_VALUE
    for (i in 0 until ring.length) {
        if (ring[i] == target) {
            val direct = Math.abs(i - atRing)
            val left = Math.min(i, atRing)
            val right = Math.max(i, atRing)
            val not_direct = left + (ring.length - right)
            val res = Math.min(direct, not_direct) + dpExplore(atKey + 1, key, i, ring, memo)
            min = Math.min(min, res)
        }
    }
    memo[atKey][atRing] = min
    return min
}

fun findMinMoves(machines: IntArray): Int {
    println(
        findMinMoves(
            intArrayOf(
                0, 3, 0
            )
        )
    )
    var sum = 0
    for (i in 0 until machines.size) {
        sum += machines[i]
    }
    if (sum % machines.size != 0) return -1
    var each = sum / machines.size

    return 0
}

class SolutionWeight(w: IntArray) {

    val treeMap = TreeMap<Int, Int>()
    var upper = 0

    init {
        var prev = 1
        for (i in 0 until w.size) {
            val weight = w[i]
            upper += w[i]
            treeMap[prev] = i
            treeMap[prev + weight - 1] = i
            prev += weight
        }
        println(treeMap)
    }

    fun pickIndex(): Int {
        val rand = Random.nextInt(1, upper + 1)
        val atIndex = treeMap[treeMap.floorKey(rand)]!!
        return atIndex!!
    }

}

fun updateBoard(board: Array<CharArray>, click: IntArray): Array<CharArray> {
    println(
        updateBoard(
            arrayOf(
                arrayOf("E", "E", "E", "E", "E").toCharArray(),
                arrayOf("E", "E", "M", "E", "E").toCharArray(),
                arrayOf("E", "E", "M", "E", "E").toCharArray(),
                arrayOf("E", "E", "E", "E", "E").toCharArray(),
                arrayOf("E", "E", "E", "E", "E").toCharArray()
            ), intArrayOf(
                3, 0
            )
        )
    )

    return arrayOf()
}

fun findSupersequence(seq1: List<Int>, seq2: List<Int>, seq3: List<Int>): List<Int> {
    val n1 = seq1.size
    val n2 = seq2.size
    val n3 = seq3.size

    // Инициализация матрицы LCS
    val lcs = Array(n1 + 1) { Array(n2 + 1) { Array(n3 + 1) { 0 } } }

    // Заполнение матрицы LCS
    for (i in 1..n1) {
        for (j in 1..n2) {
            for (k in 1..n3) {
                if (seq1[i - 1] == seq2[j - 1] && seq2[j - 1] == seq3[k - 1]) {
                    lcs[i][j][k] = lcs[i - 1][j - 1][k - 1] + 1
                } else {
                    lcs[i][j][k] = maxOf(lcs[i - 1][j][k], lcs[i][j - 1][k], lcs[i][j][k - 1])
                }
            }
        }
    }

    // Восстановление супер-последовательности из матрицы LCS
    val superseq = mutableListOf<Int>()
    var i = n1
    var j = n2
    var k = n3
    while (i > 0 && j > 0 && k > 0) {
        if (seq1[i - 1] == seq2[j - 1] && seq2[j - 1] == seq3[k - 1]) {
            superseq.add(seq1[i - 1])
            i--
            j--
            k--
        } else if (lcs[i - 1][j][k] >= lcs[i][j - 1][k] && lcs[i - 1][j][k] >= lcs[i][j][k - 1]) {
            i--
        } else if (lcs[i][j - 1][k] >= lcs[i - 1][j][k] && lcs[i][j - 1][k] >= lcs[i][j][k - 1]) {
            j--
        } else {
            k--
        }
    }

    return superseq.reversed()
}


fun threeSumMulti(arr: IntArray, target: Int): Int {
    arr.sort()
    var ans = 0L
    for (i in 0 until arr.size - 2) {
        val sum = target - arr[i]
        val res = countTwoSumCombi(arr, i + 1, arr.size - 1, sum)
        ans += res
    }
    return ans.toInt()
}

fun countTwoSumCombi(arr: IntArray, left: Int, right: Int, target: Int): Long {
    var l = left
    var r = right
    var ans = 0L
    while (l < r) {
        val curSum = arr[l] + arr[r]
        if (curSum == target) {
            var c1 = 0
            var t = arr[l]
            while (l < arr.size && arr[l] == t) {
                c1++
                l++
            }
            t = arr[r]
            var c2 = 0
            while (r >= l && arr[r] == t) {
                c2++
                r--
            }
            if (t + t == target) {
                ans += getSumOfNums(c1 - 1)
            } else ans += (c1 * c2)
        } else if (curSum < target) {
            l++
        } else {
            r--
        }
    }
    return ans
}

fun maxSumOfThreeSubarrays(nums: IntArray, k: Int): IntArray {

    val maxSumSub = IntArray(nums.size) { 0 }
    var sum = 0
    var l = 0
    for (r in 0 until nums.size) {
        sum += nums[r]
        maxSumSub[r] = sum
        if (r >= k - 1) {
            sum -= nums[l++]
        }
    }
    //starting,max
    val rightMaxSubStarting = Array(nums.size) { IntArray(0) }
    var max = 0
    var starting = 0
    for (i in nums.size - 1 downTo k) {
        if (max < maxSumSub[i]) {
            max = maxSumSub[i]
            starting = i - k + 1
        } else if (max == maxSumSub[i]) {
            starting = i - k + 1
        }
        rightMaxSubStarting[i] = intArrayOf(starting, max)
    }
    starting = 0
    max = maxSumSub[k - 1]
    var maxThreeSum = 0
    var ans = intArrayOf(0, 0, 0)
    for (j in k until nums.size - k - 1) {
        val curMiddleMax = maxSumSub[j + k - 1]
        val wholeSum = max + curMiddleMax + rightMaxSubStarting[j + k + 1][1]
        if (wholeSum > maxThreeSum) {
            maxThreeSum = wholeSum
            ans = intArrayOf(starting, j, rightMaxSubStarting[j + k + 1][0])
        }
        if (maxSumSub[j] > max) {
            max = maxSumSub[j]
            starting = j - k + 1
        }
    }
    println(ans.toList())
    return ans
}

fun threeSumMultiplicty(nums: IntArray): Int {
    println(
        threeSumMultiplicty(
            intArrayOf(

            )
        )
    )
    return 0
}

fun isScramble(s1: String, s2: String): Boolean {
    val freq1 = IntArray(26) { 0 }
    val freq2 = IntArray(26) { 0 }
    for (ch in s1) freq1[ch - 'a']++
    for (ch in s2) freq2[ch - 'a']++
    for (i in 0 until freq1.size) {
        if (freq1[i] != freq2[i]) return false
    }
    return dpExplore(s1, s2)
}

val memo = hashMapOf<String, Boolean>()

fun dpExplore(s1: String, s2: String): Boolean {
    if (s1 == s2) return true
    val key = "$s1 $s2"
    if (memo.containsKey(key)) return memo[key]!!
    var possible = false
    for (i in 1 until s1.length) {
        //not_swap
        val x1 = s1.substring(0, i)
        val y1 = s1.substring(i)
        var x2 = s2.substring(0, i)
        var y2 = s2.substring(i)
        val notSwap = dpExplore(x1, x2) && dpExplore(y1, y2)
        x2 = s2.substring(0, y1.length)
        y2 = s2.substring(y2.length)
        val swap = dpExplore(y1, x2) && dpExplore(y2, x1)
        possible = possible or (swap or notSwap)
    }
    memo[key] = possible
    return possible
}

fun superShortestCommonSubsequence(num1: IntArray, nums2: IntArray, num3: IntArray): IntArray {
    val res = dpExplore(0, 0, 0, num1, nums2, num3)
    println(res)
    return intArrayOf()
}

fun dpExplore(at1: Int, at2: Int, at3: Int, nums1: IntArray, nums2: IntArray, nums3: IntArray): MutableList<Int> {
    if (at1 == nums1.size || at2 == nums2.size || at3 == nums3.size) return mutableListOf()
    if (nums1[at1] == nums2[at2] && nums2[at2] == nums3[at3]) {
        val res = dpExplore(at1 + 1, at2 + 1, at3 + 1, nums1, nums2, nums3)
        res.add(0, nums1[at1])
        return res.toMutableList()
    }
    val res1 = dpExplore(at1 + 1, at2, at3, nums1, nums2, nums3)
    val res2 = dpExplore(at1, at2 + 1, at3, nums1, nums2, nums3)
    val res3 = dpExplore(at1, at2, at3 + 1, nums1, nums2, nums3)
    val res = listOf(res1, res2, res3).sortedWith { o1, o2 -> o2.size - o1.size }[0]
    return res.toMutableList()
}

fun removeBoxes(boxes: IntArray): Int {
    println(
        removeBoxes(
            intArrayOf(
                1, 3, 2, 2, 2, 3, 4, 3, 1
            )
        )
    )

    return 0
}

fun checkRecord(n: Int): Int {
    val memo = Array(n + 1) { Array(3) { LongArray(2) { -1L } } }
    return dpExploreAttendance(0, 0, 0, n, memo).toInt()
}

fun dpExploreAttendance(days: Int, late: Int, absent: Int, n: Int, memo: Array<Array<LongArray>>): Long {
    if (late == 3) return 0
    if (absent == 2) return 0
    if (days == n) {
        return 1
    }
    if (memo[days][late][absent] != -1L) return memo[days][late][absent]
    var cur = 0L
    cur += dpExploreAttendance(days + 1, 0, absent, n, memo)
    cur += dpExploreAttendance(days + 1, 0, absent + 1, n, memo)
    cur += dpExploreAttendance(days + 1, late + 1, absent, n, memo)
    cur %= mod
    memo[days][late][absent]
    return cur
}

fun optimalDivision(nums: IntArray): String {
    println(
        optimalDivision(
            intArrayOf(
                1000, 100, 10, 2
            )
        )
    )
    return ""
}

fun leastBricks(wall: List<List<Int>>): Int {
    val map = hashMapOf<Int, Int>()
    for (w in wall) {
        var prev = 0
        for (i in 0 until w.size) {
            prev += w[i]
            if (i != w.size - 1) map[prev] = map.getOrDefault(prev, 0) + 1
        }
    }
    return wall.size - map.values.max()
}

fun intersect(t1: Node?, t2: Node?): Node? {
    if (t1 == null || t2 == null) return null
    if (t1.isLeaf && t2.isLeaf) {
        val t1Val = if (t1.`val`) 1 else 0
        val t2Val = if (t2.`val`) 1 else 0
        val xor = t1Val xor t2Val
        return Node(xor == 1, true)
    }
    val set = mutableSetOf<Boolean>()
    val topLeft = intersect(t1.topLeft ?: t1, t2.topLeft ?: t2)
    val topRight = intersect(t1.topRight ?: t1, t2.topRight ?: t2)
    val btmLeft = intersect(t1.bottomLeft ?: t1, t2.bottomLeft ?: t2)
    val btmRight = intersect(t1.bottomRight ?: t1, t2.bottomRight ?: t2)
    set.add(topLeft?.`val` ?: false)
    set.add(topRight?.`val` ?: false)
    set.add(btmLeft?.`val` ?: false)
    set.add(btmRight?.`val` ?: false)
    if (set.size == 1 && topLeft?.isLeaf == true && topRight?.isLeaf == true && btmLeft?.isLeaf == true && btmRight?.isLeaf == true) return Node(
        set.toList()[0], true
    )
    val node = Node(false, false)
    node.topLeft = topLeft
    node.topRight = topRight
    node.bottomLeft = btmLeft
    node.bottomRight = btmRight
    return node
}

class Node(var `val`: Boolean, var isLeaf: Boolean) {
    var topLeft: Node? = null
    var topRight: Node? = null
    var bottomLeft: Node? = null
    var bottomRight: Node? = null
}

fun numSubseq(nums: IntArray, target: Int): Int {
    nums.sort()
    var ans = 0L
    val memo = hashMapOf<Int, Long>()
    var sub = 1L
    var power = 1L
    for (i in 1 until nums.size) {
        sub += power
        power *= 2
        power %= mod
        sub %= mod
        memo[i + 1] = sub
    }
    memo[1] = 1
    for (l in 0 until nums.size) {
        val rightMost = findRightMost(target - nums[l], nums)
        if (rightMost < l) break
        val res = memo[rightMost - l + 1]!!
        ans += res
        ans %= mod
    }
    return ans.toInt()
}

fun countSubsequence(range: Int, memo: HashMap<Int, Long>): Long {
    if (memo[range] != null) return memo[range]!!
    var cur = 1L
    var power = 1L
    for (i in 1 until range) {
        cur += power
        power *= 2
        power %= mod
        cur %= mod
    }
    memo[range] = cur
    return cur
}

fun findRightMost(max: Int, nums: IntArray): Int {
    var l = 0
    var r = nums.size - 1
    var ans = -1
    while (l <= r) {
        val mid = (l + r) / 2
        if (nums[mid] <= max) {
            ans = Math.max(ans, mid)
            l = mid + 1
        } else {
            r = mid - 1
        }
    }
    return ans
}


fun judgeSquareSum(c: Int): Boolean {
    for (i in 0..Math.sqrt(c.toDouble()).toInt()) {
        val sqrtB = c - (i * i)
        if (findNumSqrtBS(sqrtB)) return true
    }
    return false
}

fun findNumSqrtBS(sqrtB: Int): Boolean {
    var a = 0
    var b = sqrtB
    var max = 0
    while (a <= b) {
        val mid = a + (b - a) / 2
        if (mid * mid <= sqrtB) {
            max = Math.max(max, mid)
            a = mid + 1
        } else {
            b = mid + 1
        }
    }
    return max * max == sqrtB
}

fun judgeSquareSumSecond(c: Int): Boolean {
    if (c == 0) return true
    for (i in 0..Math.sqrt(c.toDouble()).toInt()) {
        val b = c - i * i
        if (b > 0) {
            if (b == 2) continue
            if (isSqrtNum(b)) {
                println("I $i")
                return true
            }
        }
    }
    return false
}

fun isSqrtNum(num: Int): Boolean {
    if (num == 2) return false
    val str = "${Math.sqrt(num.toDouble())}"
    if (str.substring(str.indexOf('.') + 1) == "0") {
        println("J : $str")
        return true
    }
    return false
}


fun nearestPalindromic(n: String): String {
    if (n.length == 1) {
        if (n == "0") return "1"
        return "${n.toInt() - 1}"
    } else if (StringBuilder(n).reversed().toString() == n) {

        return ""
    }
    println(
        nearestPalindromic(
            "123"
        )
    )
    return n
}


data class Fraction(val plus: Boolean, val num: Int, val divider: Int)

fun findIntegers(n: Int): Int {
    return n + 1 - dpExploreOnes(0, 0, 0, n)
}

fun dpExploreOnes(bitPos: Int, curBit: Int, facedTwoBit: Int, n: Int): Int {
    var cur = if (facedTwoBit == 1) 1 else 0
    val setOneBit = curBit or (1 shl bitPos)
    val setTwoBit = setOneBit or (1 shl bitPos + 1)
    if (setTwoBit <= n) {
        println("Two Bit : $setTwoBit")
        cur += dpExploreOnes(bitPos + 2, setTwoBit, 1, n)
    }
    if (setOneBit <= n) {
        cur += dpExploreOnes(bitPos + 1, setOneBit, facedTwoBit, n)
    }
    if (1 shl (bitPos) <= n) cur += dpExploreOnes(bitPos + 1, curBit, facedTwoBit, n)
    return cur
}

fun numDecodings(s: String): Int {
    val memo = LongArray(s.length + 1) { -1L }
    return dpExplore(0, s, memo).toInt()
}

fun dpExplore(at: Int, s: String, memo: LongArray): Long {
    if (at == s.length) return 1L
    if (s[at] == '0') return 0L
    if (memo[at] != -1L) return memo[at]
    val first = if (s[at] != '*') listOf(s[at] - '0') else (1..9).toList()
    val second = if (at + 1 < s.length) {
        if (s[at + 1] == '*') {
            (1..9).toList()
        } else listOf(s[at + 1] - '0')
    } else emptyList()
    var cur = 0L
    first.forEach {
        cur += dpExplore(at + 1, s, memo)
    }
    for (i in 0 until first.size) {
        for (j in 0 until second.size) {
            val num = first[i] * 10 + second[j]
            if (num > 26) break
            if (num in 1..26) cur += dpExplore(at + 2, s, memo)
        }
    }
    memo[at] = cur % mod
    return cur
}

class MyCircularDeque(val k: Int) {

    var front: DoubleLinked? = null
    var back: DoubleLinked? = null
    var size = 0

    fun insertFront(value: Int): Boolean {
        if (front == null || back == null) {
            size = 1
            val newNode = DoubleLinked(value)
            front = newNode
            back = newNode
            return true
        }
        if (size >= k) return false
        size++
        val newNode = DoubleLinked(value)
        front!!.prev = newNode
        newNode.next = front
        front = front!!.prev
        return true
    }

    fun insertLast(value: Int): Boolean {
        if (front == null || back == null) {
            size = 1
            val newNode = DoubleLinked(value)
            front = newNode
            back = newNode
            return true
        }
        if (size >= k) return false
        size++
        val newNode = DoubleLinked(value)
        back!!.next = newNode
        newNode.prev = back
        back = back!!.next
        return true
    }

    fun deleteFront(): Boolean {
        if (size == 0) return false
        size--
        if (front == back) {
            front = null
            back = null
            return true
        }
        front = front?.next!!
        front?.prev = null
        return true
    }

    fun deleteLast(): Boolean {
        if (size == 0) return false
        size--
        if (front == back) {
            front = null
            back = null
            return true
        }
        back = back?.prev
        back?.next = null
        return true
    }

    fun getFront(): Int {
        return front?.value ?: -1
    }

    fun getRear(): Int {
        return back?.value ?: -1
    }

    fun isEmpty(): Boolean {
        return size == 0
    }

    fun isFull(): Boolean {
        return size == k
    }

    class DoubleLinked(val value: Int) {
        var prev: DoubleLinked? = null
        var next: DoubleLinked? = null
    }

}

fun strangePrinter(s: String): Int {
    val endings = hashMapOf<Char, MutableList<Int>>()
    for (i in 0 until s.length) {
        val char = s[i]
        if (endings[char] == null) endings[char] = mutableListOf()
        endings[char]!!.add(i)
    }
    val memo = hashMapOf<String, Int>()
    return dpExplore(0, "", s, endings, memo)
}

fun dpExplore(
    at: Int, cur: String, target: String, endings: HashMap<Char, MutableList<Int>>, memo: HashMap<String, Int>
): Int {
    val key = "$at|${cur.substring(at)}"
    if (memo[key] != null) return memo[key]!!
    if (at == target.length) return 0
    if (at >= cur.length) {
        var min = Int.MAX_VALUE
        val t = target[at]
        val ends = endings[t]!!
        for (end in ends) {
            if (end >= at) {
                val repeat = (end - at + 1)
                val res = dpExplore(at + 1, cur + "$t".repeat(repeat), target, endings, memo)
                min = Math.min(min, res + 1)
            }
        }
        memo[key] = min
        return min
    }
    if (cur[at] == target[at]) return dpExplore(at + 1, cur, target, endings, memo)
    var min = Int.MAX_VALUE
    val ends = endings[target[at]]!!
    for (end in ends) {
        if (end >= at) {
            val left = cur.substring(0, at)
            val middle = "${target[at]}".repeat(end - at + 1)
            var right = ""
            if (at + (end - at + 1) <= cur.length) {
                right = cur.substring(at + (end - at + 1))
            }
            val new = left + middle + right
            val res = dpExplore(at + 1, new, target, endings, memo)
            min = Math.min(min, res + 1)
        }
    }
    memo[key] = min
    return min
}

fun checkNeighCoins(at: Int, adj: Array<MutableList<Int>>, coins: IntArray): Boolean {
    for (n in adj[at]) {
        if (coins[n] == 1) return false
    }
    return true
}

fun removeTwoAncestor(at: Int, adj: Array<MutableList<Int>>, level: Int, indegres: IntArray) {
    if (level == 0) return
    val neighs = adj[at]
    indegres[at]--
    for (n in neighs) {
        adj[n].remove(at)
        indegres[n]--
    }
    for (n in neighs) {
        removeTwoAncestor(n, adj, level - 1, indegres)
    }
}

fun minOperations(nums: IntArray, queries: IntArray): List<Long> {
    nums.sort()
    var sum = BigInteger("0")
    val prefix = Array(nums.size) { BigInteger("0") }
    for (i in 0 until nums.size) {
        sum = sum.plus(BigInteger("${nums[i]}"))
        prefix[i] = sum
    }
    val answer = mutableListOf<Long>()
    for (q in queries) {
        val atIndex = findNumber(nums, q)
        if (atIndex == -1) {
            val rightSide = prefix[prefix.size - 1].toLong() - (q * nums.size)
            answer.add(rightSide)
        } else {
            val leftSide = ((atIndex + 1) * q.toLong()) - prefix[atIndex].toLong()
            println("Q : ${(atIndex + 1) * q.toLong()}")
            val rightSide = (prefix[nums.size - 1].minus(prefix[atIndex]).toLong()) - (nums.size - atIndex - 1) * q
            println("L : $leftSide R : $rightSide")
            answer.add(leftSide + rightSide)
        }
    }
    return answer
}

fun findNumber(nums: IntArray, target: Int): Int {
    var ans = -1
    var l = 0
    var r = nums.size - 1
    while (l <= r) {
        val mid = l + (r - l) / 2
        if (nums[mid] == target) {
            return mid
        } else if (nums[mid] < target) {
            ans = Math.max(ans, mid)
            l = mid + 1
        } else r = mid - 1
    }
    return ans
}

fun primeSubOperation(nums: IntArray): Boolean {
    val primesBeka = mutableSetOf<Int>()
    primesBeka.add(2)
    for (i in 3..1000) {
        if (isPrime(i)) primesBeka.add(i)
    }
    var max = 0
    for (pri in primesBeka) {
        if (pri < nums[0]) {
            max = pri
        }
        if (pri > nums[0]) break
    }
    nums[0] -= max
    for (i in 1 until nums.size) {
        val prev = nums[i - 1]
        max = 0
        for (pri in primesBeka) {
            if (pri < nums[i] && nums[i] - pri > prev) {
                max = pri
            }
            if (pri > nums[i]) break
        }
        nums[i] -= max
    }
    for (i in 0 until nums.size - 1) {
        if (nums[i] >= nums[i + 1]) return false
    }
    return true
}

fun judgePoint24(cards: IntArray): Boolean {
    println(
        judgePoint24(
            intArrayOf(
                4, 1, 8, 7
            )
        )
    )
    return false
}

fun repeatedStringMatch(a: String, b: String): Int {

    println(
        repeatedStringMatch(
            "abcd", "cdabcdab"
        )
    )

    return 0
}

fun knightProbability(n: Int, k: Int, row: Int, column: Int): Double {
    println(
        knightProbability(
            3, 2, 0, 0
        )
    )
    return 0.0
}


fun dpExplore(at: Int, counter: Int, len: Int, nums: IntArray, memo: Array<Array<Dividing?>>): Dividing {
    if (at == nums.size || counter == 0 || nums.size - at < len) return Dividing(0, mutableListOf())
    if (memo[at][counter] != null) return memo[at][counter]!!
    var maxDiv = dpExplore(at + 1, counter, len, nums, memo)
    var sum = 0
    for (i in at until at + len) {
        sum += nums[i]
    }
    val next = dpExplore(at + len, counter - 1, len, nums, memo)
    val list = mutableListOf<Int>()
    list.add(at)
    list.addAll(next.indices)
    val curDiv = Dividing(sum + next.sum, list)
    if (curDiv.sum >= maxDiv.sum) {
        maxDiv = curDiv
    }
    memo[at][counter] = maxDiv
    return maxDiv
}

data class Dividing(val sum: Int, val indices: MutableList<Int>)

fun fallingSquares(positions: Array<IntArray>): List<Int> {

    println(
        fallingSquares(
            arrayOf(
                intArrayOf(1, 2), intArrayOf(2, 3), intArrayOf(6, 1)
            )
        )
    )

    return emptyList()
}

class Solution(val n: Int, val blacklist: IntArray) {

    val sort = blacklist.sorted().toIntArray()
    val board = mutableListOf<Int>()

    init {
        var num = 0
        var at = 0
        while (at < sort.size) {
            while (at < sort.size && num == sort[at]) {
                at++
                num++
            }
            if (num < n) board.add(num)
            num++
        }
        while (num < Math.min(n, 2 * Math.pow(10.0, 4.0).toInt())) {
            board.add(num++)
        }
    }

    fun pick(): Int {
        val rand = Random.nextInt(board.size)
        return board[rand]
    }


}

fun numSubarrayProductLessThanK(nums: IntArray, k: Int): Int {
    if (k == 0 || k == 1) return 0
    var multi = 1L
    var r = 0
    var ans = 0
    for (l in 0 until nums.size) {
        if (r < l) r = l
        while (r < nums.size && multi * nums[r] < k) {
            multi *= nums[r]
            r++
        }
        if (r - l == 0) {
            if (nums[r] < k) ans++
        } else {
            ans += r - l
        }
        if (multi != 1L) multi /= nums[l]
    }
    return ans
}

class RangeModule {

    val treeMap = TreeMap<Int, Int>()

    fun addRange(left: Int, right: Int) {
        val obj = RangeModule()
        obj.addRange(0, 9)
        obj.addRange(10, 11)
        obj.addRange(20, 23)
        obj.addRange(10, 21)
        val floor = treeMap.floorKey(left)
        var inserted = false
        if (floor != null && treeMap[floor]!! + 1 >= left) {
            treeMap[floor] = right
            inserted = true
        }
        val floor2 = treeMap.floorKey(right)
        if (floor2 != null) {
            if (inserted) {
                treeMap[floor] = Math.max(treeMap[floor2]!!, right)
            } else {
                treeMap[left] = Math.max(treeMap[floor2]!!, right)
            }
            inserted = true
        }
        if (!inserted) {
            treeMap[left] = right
        }
        println(treeMap)
    }

    fun queryRange(left: Int, right: Int): Boolean {

        return false
    }

    fun removeRange(left: Int, right: Int) {

    }

}

class DynamicSegmentTree() {

    val board = hashMapOf<Int, Int>()

    //    val max = Math.pow(10.0, 9.0).toInt() + 1
    val max = 20

    fun addQuery(l: Int, r: Int) {
        addQueryHelper(0, 0, max, l, r - 1)
    }

    private fun addQueryHelper(at: Int, l: Int, r: Int, ql: Int, qr: Int) {
        if (r < ql || qr < l) return
        else if (ql <= l && r <= qr) {
            if (board[at] != 1) board[at] = 1
        } else {
            val mid = l + (r - l) / 2
            addQueryHelper(at * 2 + 1, l, mid, ql, qr)
            addQueryHelper(at * 2 + 2, mid + 1, r, ql, qr)
        }
    }

    fun getNumsInRange(ql: Int, qr: Int): Int {
        return getNumsInRangeHelper(0, 0, max, ql, qr)
    }

    private fun getNumsInRangeHelper(at: Int, l: Int, r: Int, ql: Int, qr: Int): Int {
        if (r < ql || qr < l) return 0
        else if (ql <= l && r <= qr) {
            if (board[at] == 1) return r - l + 1
        } else {
            val mid = l + (r - l) / 2
            val left = getNumsInRangeHelper(at * 2 + 1, l, mid, ql, qr)
            val right = getNumsInRangeHelper(at * 2 + 2, mid + 1, r, ql, qr)
            return left + right
        }
        return 0
    }

}

fun smallestDistancePair(nums: IntArray, k: Int): Int {
    nums.sort()
    val mins = mutableListOf<Int>()
    var dist = 1
    while (mins.size <= k) {
        for (i in 0 until nums.size - dist) {
            val cur = nums[i]
            val next = nums[i + dist]
            val diff = next - cur
            mins.add(diff)
        }
        dist++
    }
    mins.sort()
    return mins[k - 1]
}

fun countOfAtoms(formula: String): String {
    println(
        countOfAtoms(
            "Mg(OH)2"
        )
    )
    val list = mutableListOf<Atom>()
    val map = dfsMapFreq(formula)
    println(map)
    for ((key, value) in map) list.add(Atom(key, value))
    val sorted = list.sortedWith(object : java.util.Comparator<Atom> {
        override fun compare(o1: Atom, o2: Atom): Int {
            return o1.str.compareTo(o2.str)
        }
    })
    val ans = StringBuilder()
    for (at in sorted) {
        if (at.freq == 1) ans.append(at.str)
        else ans.append(at.str).append(at.freq)
    }

    return ans.toString()
}

data class Atom(val str: String, val freq: Int)

fun dfsMapFreq(formula: String): HashMap<String, Int> {
    if (!formula.contains('(')) {
        val map = hashMapOf<String, Int>()
        var at = 0
        while (at < formula.length) {
            val chars = StringBuilder()
            chars.append(formula[at++])
            while (at < formula.length && formula[at].isLowerCase()) {
                chars.append(formula[at])
                at++
            }
            val num = java.lang.StringBuilder()
            while (at < formula.length && formula[at].isDigit()) {
                num.append(formula[at])
                at++
            }
            while (num.isEmpty()) num.append(1)
            map[chars.toString()] = num.toString().toInt()
        }
        return map
    }
    val map = hashMapOf<String, Int>()
    val stack = Stack<Char>()
    val builder = java.lang.StringBuilder()
    var at = 0
    while (at < formula.length) {
        if (formula[at] == '(') {
            builder.append('(')
            stack.push('(')
            at++
        } else if (formula[at] == ')') {
            builder.append(')')
            stack.pop()
            if (stack.isEmpty()) {
                at++
                val num = StringBuilder()
                while (at < formula.length && formula[at].isDigit()) {
                    num.append(formula[at])
                    at++
                }
                if (num.isEmpty()) num.append(1)
                val cur = dfsMapFreq(builder.substring(1, builder.length - 1))
                for ((key, value) in cur) {
                    map[key] = map.getOrDefault(key, 0) + (value * num.toString().toInt())
                }
                builder.clear()
            } else at++
        } else if (!formula[at].isDigit() && stack.isEmpty()) {
            val chars = StringBuilder()
            chars.append(formula[at++])
            while (at < formula.length && formula[at].isLowerCase()) {
                chars.append(formula[at])
                at++
            }
            val num = java.lang.StringBuilder()
            while (at < formula.length && formula[at].isDigit()) {
                num.append(formula[at])
                at++
            }
            while (num.isEmpty()) num.append(1)
            map[chars.toString()] = num.toString().toInt()
        } else if (stack.isNotEmpty()) {
            builder.append(formula[at])
            at++
        }
    }
    return map
}

fun countPalindromicSubsequences(s: String): Int {
    println(
        countPalindromicSubsequences(
            "abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba"
        )
    )
    return 0
}

fun cherryPickup(grid: Array<IntArray>): Int {
    println(
        cherryPickup(
            arrayOf(
                intArrayOf(0, 1, -1), intArrayOf(1, 0, -1), intArrayOf(1, 1, 1)
            )
        )
    )

    return 0
}

fun dpExplore(): Int {

    return 0
}

fun tilingRectangle(n: Int, m: Int): Int {
    return dpExploreReact(n, m)
}

fun dpExploreReact(n: Int, m: Int): Int {
    if (n < 0 || m < 0) return 0
    if (n == m) return 1
    val min = Int.MAX_VALUE
    var len = Math.min(n, m)
    for (i in len downTo 1) {

    }
    return 0
}

fun containVirus(isInfected: Array<IntArray>): Int {
    println(
        containVirus(
            arrayOf(
                intArrayOf(1, 1, 1, 0, 0, 0, 0, 0, 0),
                intArrayOf(1, 0, 1, 0, 1, 1, 1, 1, 1),
                intArrayOf(1, 1, 1, 0, 0, 0, 0, 0, 0)
            )
        )
    )
    val queue = LinkedList<IntArray>()
    for (i in 0 until isInfected.size) {
        for (j in 0 until isInfected[i].size) {
            if (isInfected[i][j] == 1) queue.add(intArrayOf(i, j))
        }
    }
    var ans = 0
    while (queue.isNotEmpty()) {
        var poll = queue.poll()
        if (isInfected[poll[0]][poll[1]] == -1) continue
        val cur = dfs(poll[0], poll[1], isInfected)
        println("Cur : $cur")
        ans += cur
        val size = queue.size
        for (i in 0 until size) {
            poll = queue.poll()
            if (isInfected[poll[0]][poll[1]] == -1) continue
            for (dir in dirs) {
                val nI = poll[0] + dir[0]
                val nJ = poll[1] + dir[1]
                if (nI < 0 || nJ < 0 || nI == isInfected.size || nJ == isInfected[0].size) continue
                if (isInfected[nI][nJ] == 0) {
                    isInfected[nI][nJ] = 1
                    queue.add(intArrayOf(nI, nJ))
                }
            }
        }
        for (inf in isInfected) println(inf.toList())
        println("Break   ")
    }
    return ans
}

fun dfs(i: Int, j: Int, isInfected: Array<IntArray>): Int {
    if (i < 0 || j < 0 || i == isInfected.size || j == isInfected[0].size) return 0
    if (isInfected[i][j] == 0 || isInfected[i][j] == -1) return 0
    var walls = getWalls(i, j, isInfected)
    isInfected[i][j] = -1
    walls += dfs(i - 1, j, isInfected)
    walls += dfs(i + 1, j, isInfected)
    walls += dfs(i, j - 1, isInfected)
    walls += dfs(i, j + 1, isInfected)
    return walls
}

fun getWalls(i: Int, j: Int, isInfected: Array<IntArray>): Int {
    var walls = 0
    for (dir in dirs) {
        val nI = i + dir[0]
        val nJ = j + dir[1]
        if (nI < 0 || nJ < 0 || nI == isInfected.size || nJ == isInfected[0].size) {
            continue
        }
        if (isInfected[nI][nJ] == 0) walls++
    }
    return walls
}


fun sticksVisible(n: Int, k: Int): Int {

    //1,2,3,4,5
    println(sticksVisible(5, 2))
    //1,5,4,3,2
    //2,1,5,
    return 0
}

fun globalLocalInversions(nums: IntArray): Boolean {
    println(globalLocalInversions(intArrayOf(1, 2, 3, 4, 6, 7)))
    //local nums[i]>nums[i+1]
    var local = 0
    for (i in 0 until nums.size - 1) {
        if (nums[i] > nums[i + 1]) local++
    }
    println("Local : $local")
    //global [5,4,3,1,0,2]

    return false
}

fun containsFewestViruses(r: Int, c: Int): Int {
    return dpExplore(r, c)
}

fun dpExplore(r: Int, c: Int): Int {
    if (r == 0 || c == 0) return 0
    if (r == c) return 1
    val len = Math.min(r, c)
    var min = Int.MAX_VALUE
    for (i in len downTo 1) {
        val right = dpExplore(r - i, c - i)
        val bottom = dpExplore(r - i, c)
        if (right == Int.MAX_VALUE || bottom == Int.MAX_VALUE) continue
        val ans = right + bottom + 1
        min = Math.min(min, ans)
    }
    return min
}

fun openLock(deadends: Array<String>, target: String): Int {
    val queue = LinkedList<String>()
    queue.add("0000")
    val visited = mutableSetOf<String>()
    visited.add("0000")
    var ans = 0
    deadends.forEach {
        visited.add(it)
    }
    while (queue.isNotEmpty()) {
        val size = queue.size
        for (i in 0 until size) {
            val poll = queue.poll()
            if (poll == target) return ans
            val combinations = combinations(poll)
            combinations.forEach {
                if (!visited.contains(it)) {
                    visited.add(it)
                    queue.add(it)
                }
            }
        }
        ans++
    }
    return -1
}

fun combinations(str: String): MutableList<String> {
    val list = mutableListOf<String>()
    for (i in 0 until str.length) {
        val builder = java.lang.StringBuilder(str)
        val curChar = builder[i]
        var nextChar = ((curChar - '0') + 1)
        var prevChar = ((curChar - '0') - 1)
        if (prevChar == -1) prevChar = 9
        if (nextChar == 10) nextChar = 0
        builder.setCharAt(i, "$nextChar"[0])
        list.add(builder.toString())
        builder.setCharAt(i, "$prevChar"[0])
        list.add(builder.toString())
    }
    return list
}

fun crackSafe(n: Int, k: Int): String {
    println(
        crackSafe(
            2, 2
        )
    )


    return ""
}

fun reachNumber(target: Int): Int {
    println(
        reachNumber(
            2
        )
    )
    return 0
}

fun makeLargestSpecial(s: String): String {
    println(
        makeLargestSpecial(
            "11011000"
        )
    )
    return s
}

fun minSwapsCouples(row: IntArray): Int {
    return dfsCouples(0, row)
}

fun dfsCouples(at: Int, row: IntArray): Int {
    if (at >= row.size) return 0
    if (row[at] % 2 == 0) {
        val next = row[at] + 1
        var swap = -1
        for (i in at + 1 until row.size) {
            if (row[i] == next) {
                swap = i
                break
            }
        }
        val temp = row[at + 1]
        row[at + 1] = row[swap]
        row[swap] = temp
        return (if (at + 1 == swap) 0 else 1) + dfsCouples(at + 2, row)
    } else {
        val next = row[at] - 1
        var swap = -1
        for (i in at + 1 until row.size) {
            if (row[i] == next) {
                swap = i
                break
            }
        }
        val temp = row[at + 1]
        row[at + 1] = row[swap]
        row[swap] = temp
        return (if (at + 1 == swap) 0 else 1) + dfsCouples(at + 2, row)
    }
}

fun orderOfLargestPlusSign(n: Int, mines: Array<IntArray>): Int {
    if (n == 1) {
        if (mines.isNotEmpty()) return 0
        return 1
    }
    val init = Array(n) { IntArray(n) { 1 } }
    val left = Array(n) { IntArray(n) { 1 } }
    val top = Array(n) { IntArray(n) { 1 } }
    val right = Array(n) { IntArray(n) { 1 } }
    val bottom = Array(n) { IntArray(n) { 1 } }
    markIt(init, mines)
    markIt(left, mines)
    markIt(right, mines)
    markIt(top, mines)
    markIt(bottom, mines)
    for (i in 0 until n) {
        var s = 0
        for (j in 0 until n) {
            if (left[i][j] == 0) s = 0
            s += left[i][j]
            left[i][j] = s
        }
    }
    for (i in n - 1 downTo 0) {
        var s = 0
        for (j in n - 1 downTo 0) {
            if (right[i][j] == 0) s = 0
            s += right[i][j]
            right[i][j] = s
        }
    }
    for (j in 0 until n) {
        var s = 0
        for (i in 0 until n) {
            if (top[i][j] == 0) s = 0
            s += top[i][j]
            top[i][j] = s
        }
    }
    for (j in n - 1 downTo 0) {
        var s = 0
        for (i in n - 1 downTo 0) {
            if (bottom[i][j] == 0) s = 0
            s += bottom[i][j]
            bottom[i][j] = s
        }
    }
    var ans = 0
    for (i in 1 until n - 1) {
        for (j in 1 until n - 1) {
            if (init[i][j] == 1) {
                var min = Int.MAX_VALUE
                min = Math.min(min, left[i][j - 1])
                min = Math.min(min, right[i][j + 1])
                min = Math.min(min, top[i - 1][j])
                min = Math.min(min, bottom[i][j + 1])
                ans = Math.max(ans, min)
            }
        }
    }
    return ans
}

fun markIt(b: Array<IntArray>, mines: Array<IntArray>) {
    for ((i, j) in mines) {
        b[i][j] = 0
    }
}

fun dfs(at: Int, row: IntArray): Int {
    if (at >= row.size) return 0
    if (Math.abs(row[at] - row[at + 1]) == 1) return dfs(at + 2, row)

    if (row[at] % 2 == 0) {
        val next = row[at] + 1
        val clone = row.clone()
        var change = -1
        for (i in at + 2 until row.size) {
            if (row[i] == next) {
                change = i
                break
            }
        }

    }
    return 0
}


fun basicCalculatorIV(expression: String, evalvars: Array<String>, evalints: IntArray): List<String> {

    return emptyList()
}

fun findSmallestInteger(nums: IntArray, value: Int): Int {
    val board = IntArray(nums.size) { 0 }
    for (i in 0 until nums.size) {
        board[i] = nums[i] % value
        if (board[i] < 0) board[i] += value
    }
    val freq = hashMapOf<Int, Int>()
    for (b in board) freq[b] = freq.getOrDefault(b, 0) + 1
    for (ans in 0 until nums.size) {
        val mod = ans % value
        val fr = freq.getOrDefault(mod, 0)
        if (fr == 0) return ans
        freq[mod] = freq.getOrDefault(mod, 0) - 1
    }
    return nums.size
}

fun beautifulSubsets(nums: IntArray, k: Int): Int {
    return dfs(0, nums, hashMapOf(), k)
}

fun dfs(at: Int, nums: IntArray, map: HashMap<Int, Int>, k: Int): Int {
    if (at == nums.size) {
        if (map.size != 0) {
            println(map)
            return 1
        }
        return 0
    }
    var c = 0
    if (!map.contains(nums[at] - k)) {
        map[nums[at]] = map.getOrDefault(nums[at], 0) + 1
        c += dfs(at + 1, nums, map, k)
        map[nums[at]] = map.getOrDefault(nums[at], 0) - 1
        if (map[nums[at]] == 0) map.remove(nums[at])
    }
    c += dfs(at + 1, nums, map, k)
    return c
}

fun checkValidGrid(grid: Array<IntArray>): Boolean {
    var i = 0
    var j = 0
    var at = 0
    while (at < grid.size * grid.size) {
        val res = checkMove(i, j, grid, at + 1)
        if (res.i != -1) {
            if (at + 1 == grid.size * grid.size - 1) return true
            i = res.i
            j = res.j
            at++
        } else return false
    }
    return true
}

val dirsK = arrayOf(
    intArrayOf(-1, -2),
    intArrayOf(-2, -1),
    intArrayOf(-2, 1),
    intArrayOf(-1, 2),
    intArrayOf(1, -2),
    intArrayOf(2, -1),
    intArrayOf(2, 1),
    intArrayOf(1, 2)
)

fun checkMove(i: Int, j: Int, grid: Array<IntArray>, t: Int): Knight {

    for (dir in dirsK) {
        val x = i + dir[0]
        val y = j + dir[1]
        if (x < 0 || y < 0 || x >= grid.size || y >= grid.size) continue
        if (grid[x][y] == t) return Knight(x, y)
    }
    return Knight(-1, -1)
}

data class Knight(val i: Int, val j: Int)

fun evenOddBit(n: Int): IntArray {
    val bin = Integer.toBinaryString(n).reversed()
    var odd = 0
    var even = 0
    for (i in 0 until bin.length) {
        if (i % 2 == 0 && bin[i] == '1') {
            even++
        } else if (i % 2 == 1 && bin[i] == '1') {
            odd++
        }
    }
    return intArrayOf(even, odd)
}

fun reachingPoints(sx: Int, sy: Int, tx: Int, ty: Int): Boolean {
    println(
        reachingPoints(
            1, 1, 3, 5
        )
    )
    if (setOf(sx, sy, tx, ty).size == 1) return true
    if (tx == ty) return false
    return dfs(tx, ty, sx, sy)
}

fun dfs(tx: Int, ty: Int, sx: Int, sy: Int): Boolean {
    if (tx == sx && ty == sy) return true
    if (tx < sx || ty < sy) return false
    val min = Math.min(tx, ty)
    return dfs(tx, ty - min, sx, sy) || dfs(tx - min, ty, sx, sy)
}

fun movesToChessboard(board: Array<IntArray>): Int {
    println(
        movesToChessboard(
            arrayOf(
                intArrayOf(0, 1, 1, 0), intArrayOf(0, 1, 1, 0), intArrayOf(1, 0, 0, 1), intArrayOf(1, 0, 0, 1)
            )
        )
    )
    return 0
}

fun kthSmallestPrimeFraction(arr: IntArray, k: Int): IntArray {
    val queue = PriorityQueue(object : Comparator<MaxDouble> {
        override fun compare(o1: MaxDouble, o2: MaxDouble): Int {
            if (o1.divided <= o2.divided) return 1
            return -1
        }
    })
    for (i in 0 until arr.size) {
        for (j in i + 1 until arr.size) {
            queue.add(MaxDouble(arr[i], arr[j], arr[i].toDouble() / arr[j].toDouble()))
            if (queue.size > k) queue.poll()
        }
    }
    return intArrayOf(queue.peek().first, queue.poll().second)
}

data class MaxDouble(val first: Int, val second: Int, val divided: Double)

fun rotatedDigits(n: Int): Int {
    val nums = intArrayOf(0, 1, 8, 2, 5, 6, 9)
    var res = 0
    for (i in 1 until nums.size) res += dpExplore(nums, nums[i], n)
    return res
}

fun dpExplore(nums: IntArray, cur: Int, max: Int): Int {
    if (cur > max) return 0
    var c = if (rotateChange(cur)) {
        1
    } else 0
    for (s in nums) {
        c += dpExplore(nums, cur * 10 + s, max)
    }
    return c
}

fun rotateChange(num: Int): Boolean {
    val set = setOf('2', '5', '6', '9')
    set.forEach {
        if ("$num".contains(it)) return true
    }
    return false
}

fun canPlaceFlowers(flowerbed: IntArray, n: Int): Boolean {
    var l = 0
    var zeroes = 0
    while (l < flowerbed.size && flowerbed[l] == 0) {
        l++
        zeroes++
    }
    var ans = zeroes / 2
    if (l == flowerbed.size) {
        if (zeroes % 2 == 1) ans++
        return ans >= n
    }
    var r = flowerbed.size - 1
    zeroes = 0
    while (r >= 0 && flowerbed[r] == 0) {
        zeroes++
        r--
    }
    ans += zeroes / 2
    zeroes = 0
    for (i in l..r) {
        if (flowerbed[i] == 0) {
            zeroes++
        } else {
            ans += getFlowers(zeroes)
            zeroes = 0
        }
    }
    return ans >= n
}

fun getFlowers(zeroes: Int): Int {
    var c = 0
    var zero = zeroes - 2
    while (zero > 0) {
        c++
        zero -= 2
    }
    return c
}


fun preimageSizeFZF(k: Int): Int {
    println(
        preimageSizeFZF(
            3
        )
    )
    return 0
}

fun spreadingDisease(adj: Array<IntArray>, viruses: IntArray): Int {

    println(
        spreadingDisease(
            arrayOf(
                intArrayOf(1, 2), intArrayOf(2, 3), intArrayOf(5, 6, 7), intArrayOf(1, 9, 10)
            ), intArrayOf(
                1, 3, 4, 5, 6, 10
            )
        )
    )

    val group = IntArray(adj.size) { -1 }
    var component = 1
    val componentSize = hashMapOf<Int, Int>()
    for (i in 0 until viruses.size) {
        if (group[i] != -1) {
            componentSize.remove(group[i])
            continue
        }
        val sizeComponent = calculateSize(adj, viruses[i], component, group)
        componentSize[component] = sizeComponent
        component++
    }
    var max = 0
    var ans = 0
    for ((comp, size) in componentSize) {
        if (size > max) {
            max = size
            ans = comp
        } else if (size == max && comp < ans) {
            ans = comp
        }
    }
    return ans
}


fun countUniqueCharactersOfAllSubstrings(st: String): Int {


    return 0
}


fun calculateSize(adj: Array<IntArray>, at: Int, component: Int, group: IntArray): Int {
    var c = 1
    group[at] = component
    for (n in adj[at]) {
        if (group[n] == -1) {
            c += calculateSize(adj, n, component, group)
        }
    }
    return c
}

fun busRoutes(routes: Array<IntArray>, s: Int, t: Int): Int {
    println(
        busRoutes(
            arrayOf(
                intArrayOf(1, 3, 4, 6, 7, 8, 10), intArrayOf(2, 5, 6, 8, 10, 16), intArrayOf(10, 14, 17)
            ), 2, 17
        )
    )
    val adj = Array(routes.size) { mutableListOf<Int>() }
    routes.forEach { it.sort() }
    val queue = LinkedList<IntArray>()
    val targets = mutableSetOf<Int>()
    val visited = mutableSetOf<Int>()
    for (i in 0 until routes.size) {
        if (Arrays.binarySearch(routes[i], s) >= 0) {
            queue.add(intArrayOf(i, 1))
            visited.add(i)
        }
        if (Arrays.binarySearch(routes[i], t) >= 0) {
            targets.add(i)
        }
    }
    for (i in 0 until routes.size) {
        for (j in i + 1 until routes.size) {
            if (shareCommon(routes[i], routes[j])) {
                adj[i].add(j)
                adj[j].add(i)
            }
        }
    }
    while (queue.isNotEmpty()) {
        val size = queue.size
        for (j in 0 until size) {
            val poll = queue.poll()
            if (targets.contains(poll[0])) return poll[1]
            for (nei in adj[poll[0]]) {
                if (!visited.contains(nei)) {
                    visited.add(nei)
                    queue.add(intArrayOf(nei, poll[1] + 1))
                }
            }
        }
    }
    return -1
}

fun shareCommon(route1: IntArray, route2: IntArray): Boolean {
    var a = 0
    var b = 0
    while (a < route1.size && b < route2.size) {
        if (route1[a] == route2[b]) return true
        else if (route1[a] < route2[b]) a++
        else b++
    }
    return false
}

fun bfsReachedTop(r: Int, c: Int, seen: MutableSet<Cell>, grid: Array<IntArray>): Boolean {
    val queue = LinkedList<Cell>()
    queue.add(Cell(r, c))
    seen.add(Cell(r, c))
    if (r == 0) return true
    while (queue.isNotEmpty()) {
        val size = queue.size
        for (i in 0 until size) {
            val poll = queue.poll()
            for (dir in dirs) {
                val nI = poll.r + dir[0]
                val nJ = poll.c + dir[1]
                if (nI < 0 || nJ < 0 || nI == grid.size || nJ == grid[0].size || grid[nI][nJ] == 0 || seen.contains(
                        Cell(
                            nI, nJ
                        )
                    )
                ) continue
                if (nI == 0) return true
                seen.add(Cell(nI, nJ))
                queue.add(Cell(nI, nJ))
            }
        }
    }
    return false
}

data class Cell(val r: Int, val c: Int)

fun repairCars(ranks: IntArray, cars: Int): Long {

    var l = 0L
    var r = Long.MAX_VALUE
    var ans = Long.MAX_VALUE
    while (l <= r) {
        val mid = l + (r - l) / 2
        if (isPossible(mid, ranks, cars)) {
            ans = Math.min(ans, mid)
            r = mid - 1
        } else l = mid + 1
    }
    return ans
}

fun isPossible(curMin: Long, ranks: IntArray, maxCars: Int): Boolean {
    var cur = 0L
    for (r in ranks) {
        val power = curMin / r
        val repair = Math.sqrt(power.toDouble()).toInt()
        cur += repair
    }
    return cur >= maxCars
}


fun findScore(nums: IntArray): Long {

    val queue = PriorityQueue(object : Comparator<Int> {
        override fun compare(o1: Int, o2: Int): Int {
            if (nums[o1] == nums[o2]) return o1 - o2
            return nums[o1] - nums[o2]
        }
    })
    for (i in 0 until nums.size) {
        queue.add(i)
    }
    var score = 0L
    val marked = Array(nums.size) { false }
    while (queue.isNotEmpty()) {
        val poll = queue.poll()
        if (marked[poll]) continue
        score += nums[poll]
        marked[poll] = true
        if (poll - 1 >= 0) marked[poll - 1] = true
        if (poll + 1 < nums.size) marked[poll + 1] = true
    }
    return score
}

fun maximizeGreatness(nums: IntArray): Int {
    val map = TreeMap<Int, Int>()
    for (n in nums) {
        map[n] = map.getOrDefault(n, 0) + 1
    }
    val sorted = nums.sorted()
    var ans = 0
    for (n in sorted) {
        if (map[n] == 0) continue
        if (map.ceilingKey(n + 1) != null) {
            if (map[map.ceilingKey(n + 1)]!! > 0) {
                map[map.ceilingKey(n + 1)] = map[map.ceilingKey(n + 1)]!! - 1
                ans++
                if (map[map.ceilingKey(n + 1)] == 0) map.remove(map.ceilingKey(n + 1))
            } else break
        }
    }
    return ans
}

fun distMoney(money: Int, children: Int): Int {
    if (money < children) return -1
    else if (money == children) return 0
    return recursionBeksultan(money, children)
}

fun recursionBeksultan(money: Int, chil: Int): Int {
    if (chil == 2 && money == 12) return 0
    if (chil == 0) return if (money > 0) -1 else 0
    if (money <= chil) return 0
    if (money - 8 >= (chil - 1) && chil > 0) {
        return 1 + recursionBeksultan(money - 8, chil - 1)
    } else return 0
}

fun numSubarrayBoundedMax(nums: IntArray, left: Int, right: Int): Int {
    println(
        numSubarrayBoundedMax(
            intArrayOf(
                2, 1, 2, 4
            ), 1, 2
        )
    )
    val lEffect = IntArray(nums.size) { 0 }
    val rEffect = IntArray(nums.size) { nums.size - 1 }
    val stack = Stack<Int>()
    for (i in 0 until nums.size) {
        while (stack.isNotEmpty() && nums[stack.peek()] <= nums[i]) {
            stack.pop()
        }
        if (stack.isNotEmpty()) {
            lEffect[i] = stack.peek() + 1
        }
        stack.push(i)
    }
    stack.clear()
    for (i in nums.size - 1 downTo 0) {
        while (stack.isNotEmpty() && nums[stack.peek()] <= nums[i]) {
            stack.pop()
        }
        if (stack.isNotEmpty()) {
            rEffect[i] = stack.peek() - 1
        }
        stack.push(i)
    }
    var ans = 0
    for (i in 0 until nums.size) {
        val cur = nums[i]
        if (cur in left..right) {
            val l = lEffect[i]
            val r = rEffect[i]
            ans += (i - l + 1) * (r - i + 1)
        }
    }
    return ans
}

fun bestRotation(nums: IntArray): Int {

    println(
        bestRotation(
            intArrayOf(
                1, 3, 0, 2, 4
            )
        )
    )

    return 0
}

fun minSwap(nums1: IntArray, nums2: IntArray): Int {
    println(
        minSwap(
            intArrayOf(
                0, 3, 5, 8, 9
            ), intArrayOf(
                2, 1, 4, 6, 9
            )
        )
    )
    return dpExplore(0, 0, nums1, nums2)
}

fun dpExplore(at1: Int, at2: Int, nums1: IntArray, nums2: IntArray): Int {

    return 0
}

fun splitArraySameAverage(nums: IntArray): Boolean {
    println(
        splitArraySameAverage(
            intArrayOf(
                6, 8, 18, 3, 1
            )
        )
    )
    val sum = nums.sum()
    val memo = hashMapOf<String, Boolean>()
    return dpExplore(0, 0, 0, 0, nums, sum, memo)
}

fun dpExplore(
    at: Int, bit: Int, curSum: Int, counter: Int, nums: IntArray, totalSum: Int, memo: HashMap<String, Boolean>
): Boolean {
    if (at == nums.size) return false
    val ave1 = (curSum) / counter.toDouble()
    val ave2 = (totalSum - curSum) / (nums.size - counter).toDouble()
    if (ave1 == ave2) {
        return true
    }
    val key = "$at $bit"
    if (memo.containsKey(key)) return memo[key]!!
    for (i in 0 until nums.size) {
        if (bit and (1 shl i) == 0) {
            val newBit = bit or (1 shl i)
            if (dpExplore(at + 1, newBit, curSum + nums[i], counter + 1, nums, totalSum, memo)) return true
        }
    }
    memo[key] = false
    return false
}

fun xorGame(nums: IntArray): Boolean {
    println(
        xorGame(
            intArrayOf(
                1, 2, 3
            )
        )
    )
    return false
}

fun largestTriangleArea(points: Array<IntArray>): Double {
    var ans = 0.0
    for (i in 0 until points.size) {
        for (j in i + 1 until points.size) {
            for (k in j + 1 until points.size) {
                val res = calculateArea(points[i], points[j], points[k])
                ans = Math.max(res, ans)
            }
        }
    }
    return ans
}

fun calculateArea(p1: IntArray, p2: IntArray, p3: IntArray): Double {
    val xs = (arrayOf(p1, p2, p3)).map { it[0] }.toSet()
    val ys = (arrayOf(p1, p2, p3)).map { it[1] }.toSet()
    if (xs.size == 1 || ys.size == 1) return 0.0
    val res = (p1[0] * (p2[1] - p3[1])) + (p2[0] * (p3[1] - p1[1])) + (p3[0] * (p1[1] - p2[1]))
    return Math.abs(res) / 2.0
}

fun racecar(target: Int): Int {
    val visited = mutableSetOf<String>()
    val queue = LinkedList<CarState>()
    queue.add(CarState("", 0, 1))
    while (queue.isNotEmpty()) {
        val size = queue.size
        for (i in 0 until size) {
            val poll = queue.poll()
            if (poll.pos == target) {
                return poll.path.length
            }
            //A
            var nSpeed = poll.speed * 2
            val nPos = poll.pos + poll.speed
            if (!visited.contains("$nPos|$nSpeed")) {
                visited.add("$nPos|$nSpeed")
                queue.add(CarState(poll.path + "A", nPos, nSpeed))
            }
            //R
            nSpeed = if (poll.speed > 0) -1 else 1
            if (!visited.contains("${poll.pos}|${nSpeed}")) {
                visited.add("${poll.pos}|${nSpeed}")
                queue.add(CarState(poll.path + "R", poll.pos, nSpeed))
            }
        }
    }
    return -1
}

data class CarState(val path: String, val pos: Int, val speed: Int)

fun numBusesToDestination(routes: Array<IntArray>, source: Int, target: Int): Int {

    val adj = hashMapOf<Int, MutableSet<Dest>>()
    var ans = Int.MAX_VALUE
    for (i in 0 until routes.size) {
        val route = routes[i]
        for (j in 0 until route.size) {
            val prev = route[j]
            val next = if (j + 1 == route.size) route[0] else route[j + 1]
            if (adj[prev] == null) {
                adj[prev] = mutableSetOf()
            }
            adj[prev]!!.add(Dest(next, i))
        }
    }
    val queue = PriorityQueue(object : Comparator<BusStopState> {
        override fun compare(o1: BusStopState, o2: BusStopState): Int {
            return o1.difBus - o2.difBus
        }
    })
    val visited = mutableSetOf<SeenBus>()
    queue.add(BusStopState(source, -1, 0))
    while (queue.isNotEmpty()) {
        val size = queue.size
        for (i in 0 until size) {
            val poll = queue.poll()
            if (poll.route == target) {
                return poll.difBus
            }
            visited.add(SeenBus(poll.route, poll.atBus))
            val atRoute = poll.route
            for (n in adj[atRoute] ?: mutableSetOf()) {
                if (poll.atBus == -1 && !visited.contains(SeenBus(n.route, n.bus))) {
                    queue.add(BusStopState(n.route, n.bus, 1))
                } else if (!visited.contains(SeenBus(n.route, n.bus))) {
                    queue.add(BusStopState(n.route, n.bus, poll.difBus + if (n.bus != poll.atBus) 1 else 0))
                }
            }
        }
    }
    return -1
}

data class BusStopState(val route: Int, val atBus: Int, val difBus: Int)

data class Dest(val route: Int, val bus: Int)

data class SeenBus(val route: Int, val atBus: Int)


fun numFriendRequests(ages: IntArray): Int {
    println(
        numFriendRequests(
            intArrayOf(
                20, 30, 100, 110, 120
            )
        )
    )
    val sorted = ages.toSet().sorted().toIntArray()
    val freq = hashMapOf<Int, Int>()
    for (age in ages) freq[age] = freq.getOrDefault(age, 0) + 1
    var ans = 0
    for ((key, value) in freq) {
        if (value > 1) {
            ans += (value * (value - 1))
        }
    }
    for (i in 0 until sorted.size) {
        val leftMost = leftMostIndexValue(sorted[i] / 2 + 7, i, sorted)
        ans += (i - leftMost)
    }
    return ans
}

fun leftMostIndexValue(min: Int, at: Int, sorted: IntArray): Int {
    var l = 0
    var r = at
    var ans = r
    while (l <= r) {
        val index = (l + r) / 2
        val value = sorted[index]
        if (value > min) {
            ans = Math.min(ans, index)
            r = index - 1
        } else l = index + 1
    }
    return ans
}

fun consecutiveNumbersSum(n: Int): Int {


    println(
        consecutiveNumbersSum(
            15
        )
    )
    return 0
}

fun isRectangleOverlap(rec1: IntArray, rec2: IntArray): Boolean {

    val sortByX = arrayOf(rec1, rec2).sortedWith(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            if (o1[0] == o2[0]) return o1[2] - o2[2]
            return o1[0] - o2[0]
        }
    })
    val sortByY = arrayOf(rec1, rec2).sortedWith(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            if (o1[1] == o2[1]) return o1[2] - o2[2]
            return o1[1] - o2[1]
        }
    })
    if (sortByX[1][0] in sortByX[0][0]..sortByX[0][2]) return true
    if (sortByY[1][1] in sortByY[0][1]..sortByY[0][3]) return true
    return false
}

fun shortestPathLength(graph: Array<IntArray>): Int {

    println(
        shortestPathLength(
            arrayOf(
                intArrayOf(), intArrayOf(), intArrayOf()
            )
        )
    )

    return 0
}


data class Point(val x: Int, val y: Int)


fun kSimilarity(s1: String, s2: String): Int {

    if (s1 == s2) return 0
    val queue = LinkedList<StrState>()
    queue.add(StrState(StringBuilder(s1), 0, 0))
    while (queue.isNotEmpty()) {
        val size = queue.size
        for (i in 0 until size) {
            val poll = queue.poll()
            var at = poll.at
            while (at < s2.length && poll.builder[at] == s2[at]) {
                at++
            }
            if (at == s2.length) return poll.swaps
            val curChar = poll.builder[at]
            val targetChar = s2[at]
            var atIndexTarget = mutableListOf<Int>()
            var found = false
            for (j in at until s2.length) {
                if (poll.builder[j] == targetChar && s2[j] == curChar) {
                    poll.builder[j] = s2[j]
                    found = true
                    break
                }
                if (poll.builder[j] == targetChar) {
                    atIndexTarget.add(j)
                }
            }
            if (found) {
                queue.add(StrState(poll.builder, at + 1, poll.swaps + 1))
            } else {
                for (t in atIndexTarget) {
                    val newBuilder = StringBuilder(poll.builder)
                    newBuilder.setCharAt(t, curChar)
                    queue.add(StrState(newBuilder, at + 1, poll.swaps + 1))
                }
            }
        }
    }
    return -1
}

data class StrState(val builder: StringBuilder, val at: Int, val swaps: Int)

class ExamRoom(val n: Int) {

    init {
        val obj = ExamRoom(10)
        println(obj.seat())
    }

    val treeSet = TreeSet<Int>()


    fun seat(): Int {
        return 0
    }

    fun leave(p: Int) {

    }

}

fun shortestSubarray(nums: IntArray, k: Int): Int {

    println(
        shortestSubarray(
            intArrayOf(2, -1, 2), 3
        )
    )
    return 0
}

fun shortestPathAllKeys(grid: Array<String>): Int {
    var bit = 0
    var start = intArrayOf()
    val keyLockPos = hashMapOf<Char, Int>()
    for (i in 0 until grid.size) {
        for (j in 0 until grid[i].length) {
            if (grid[i][j] == '@') {
                start = intArrayOf(i, j)
            } else if (grid[i][j] != '.' && grid[i][j] != '#') {
                val keyOrLock = grid[i][j]
                if (!keyLockPos.containsKey(keyOrLock.toLowerCase())) {
                    keyLockPos[keyOrLock.toLowerCase()] = bit
                    keyLockPos[keyOrLock.toUpperCase()] = bit
                    bit++
                }
            }
        }
    }
    var targetBit = 0
    for (i in 0 until keyLockPos.size / 2) {
        targetBit = targetBit or (1 shl i)
    }
    println("T : $targetBit, KeyLockPos : $keyLockPos start : ${start.toList()}")
    println(bfs(2, keyLockPos, grid, 2, 0).toList())
    val res = dfs(start[0], start[1], 0, targetBit, grid, keyLockPos)
    if (res == Int.MAX_VALUE) return -1
    return res
}

fun dfs(i: Int, j: Int, bit: Int, targetBit: Int, grid: Array<String>, keyLockPos: HashMap<Char, Int>): Int {
    if (targetBit == bit) return 0
    var moves = Int.MAX_VALUE
    for (b in 0 until keyLockPos.size / 2) {
        if (bit and (1 shl b) == 0) {
            val res = bfs(bit, keyLockPos, grid, i, j)
            if (res.isNotEmpty()) {
                println("Res : ${res.toList()} when checking : $b")
                val cur = dfs(res[2], res[3], res[1], targetBit, grid, keyLockPos)
                if (cur != Int.MAX_VALUE) {
                    moves = Math.min(moves, cur + res[0])
                }
            }
        }
    }
    return moves
}

fun bfs(
    keyBit: Int, keyLockPos: HashMap<Char, Int>, grid: Array<String>, i: Int, j: Int
): IntArray {
    var c = 0
    val queue = LinkedList<IntArray>()
    queue.add(intArrayOf(i, j))
    val dirs = arrayOf(
        intArrayOf(1, 0), intArrayOf(-1, 0), intArrayOf(0, 1), intArrayOf(0, -1)
    )
    val visited = Array(grid.size) { BooleanArray(grid[0].length) { false } }
    visited[i][j] = true
    while (queue.isNotEmpty()) {
        val size = queue.size
        for (s in 0 until size) {
            val poll = queue.poll()
            val value = grid[poll[0]][poll[1]]
            if (value.isLowerCase() && (keyBit and (1 shl keyLockPos[value.toLowerCase()]!!) == 0)) {
                val bitPos = keyLockPos[value]!!
                val newBit = keyBit or (1 shl bitPos)
                return intArrayOf(c, newBit, poll[0], poll[1])
            }
            for (dir in dirs) {
                val x = poll[0] + dir[0]
                val y = poll[1] + dir[1]
                if (x < 0 || y < 0 || x == grid.size || y == grid[0].length) continue
                else if (grid[x][y] == '#') continue
                if (grid[x][y].isUpperCase()) {
                    val bitPos = keyLockPos[grid[x][y]]!!
                    if (keyBit and (1 shl bitPos) != 0 && !visited[x][y]) {
                        queue.add(intArrayOf(x, y))
                        visited[x][y] = true
                    }
                } else if (grid[x][y] == '@' || grid[x][y] == '.' || grid[x][y].isLowerCase()) {
                    if (!visited[x][y]) {
                        queue.add(intArrayOf(x, y))
                        visited[x][y] = true
                    }
                }
            }
        }
        c++
    }
    return intArrayOf()
}


fun primePalindrome(n: Int): Int {
    dfs("", n)
    for (i in 0..9) {
        dfs("$i", n)
    }
    return ans
}

var ans = Int.MAX_VALUE
fun dfs(cur: String, bound: Int) {
    if (cur.length > 9) return
    if (cur.toIntOrNull() != null && cur.toInt() >= bound && isPrime(cur.toInt())) {
        ans = Math.min(ans, cur.toInt())
    }
    for (i in 0..9) {
        val newStr = "$i$cur$i"
        dfs(newStr, bound)
    }
}


fun profitableSchemes(n: Int, minProfit: Int, group: IntArray, profit: IntArray): Int {
    val memo = Array(n + 1) { Array(group.size) { LongArray(minProfit + 1) { -1L } } }
    val res = dpExplore(n, 0, 0, group, profit, minProfit, memo).toInt()
    return res
}

fun dpExplore(
    n: Int, at: Int, curProfit: Int, group: IntArray, profit: IntArray, minProfit: Int, memo: Array<Array<LongArray>>
): Long {
    if (at == group.size) return 0L
    val min = Math.min(curProfit, minProfit)
    if (memo[n][at][min] != -1L) return memo[n][at][minProfit]
    var c = dpExplore(n, at + 1, curProfit, group, profit, minProfit, memo)
    if (group[at] <= n) {
        val newProfit = curProfit + profit[at]
        if (newProfit >= minProfit) {
            c++
        }
        c += dpExplore(n - group[at], at + 1, newProfit, group, profit, minProfit, memo)
    }
    memo[n][at][minProfit] = c % mod
    return c % mod
}

fun decodeAtIndex(s: String, k: Int): String {
    println(
        decodeAtIndex(
            "leet2code3", 10
        )
    )
    return s
}

fun sumSubseqWidths(nums: IntArray): Int {
    println(
        sumSubseqWidths(
            intArrayOf(
                0, 3, 1, 6, 2, 2, 7
            )
        )
    )
    return 0
}

fun allPossibleFBT(n: Int): List<TreeNode?> {
    if (n % 2 == 0) return emptyList()
    println(
        allPossibleFBT(
            7
        )
    )
    return emptyList()
}

fun subarrayBitwiseORs(arr: IntArray): Int {

    println(
        subarrayBitwiseORs(
            intArrayOf(
                1, 1, 2
            )
        )
    )

    return 0
}

fun atMostNGivenDigitSet(digits: Array<String>, n: Int): Int {
    return dpExplore(0, digits, n) - 1
}

fun dpExplore(curNum: Long, digits: Array<String>, n: Int): Int {
    if (curNum > n) return 0
    var c = 1
    for (dig in digits) {
        val newNum = curNum * 10L + (dig[0] - '0')
        c += dpExplore(newNum, digits, n)
    }
    return c
}

fun numPermsDISequence(s: String): Int {


    println(
        numPermsDISequence(
            "DID"
        )
    )
    return 0
}

fun superpalindromesInRange(left: String, right: String): Int {
    var c = dfs("", left.toLong(), right.toLong())
    for (i in 0..9) {
        val res = dfs("$i", left.toLong(), right.toLong())
        c += res
    }
    return c
}

fun dfs(cur: String, left: Long, right: Long): Int {
    if (cur.length >= 10) return 0
    var c = 0
    if (cur.toLongOrNull() != null && cur[0] != '0') {
        val multi = cur.toLong() * cur.toLong()
        if (multi in left..right && isPalindrome(multi)) {
            println("cur : $cur")
            c++
        }
    }
    for (i in 0..9) {
        val pal = "$i$cur$i"
        c += dfs(pal, left, right)
    }
    return c
}

fun isPalindrome(multi: Long): Boolean {
    var a = 0
    val str = "$multi"
    var b = str.length - 1
    while (a < b) {
        if (str[a] != str[b]) return false
        a++
        b--
    }
    return true
}


fun smallestRangeII(nums: IntArray, k: Int): Int {
    if (nums.size == 1) return 0
    val sorted = nums.sorted()
    var dif = sorted[sorted.size - 1] - sorted[0]
    if (dif <= k) return dif
    val plus = sorted[0] + k
    val minus = sorted[sorted.size - 1] - k
    var min = Math.min(plus, minus)
    var max = Math.max(plus, minus)
    dif = max - min
    nums.sort()
//    println("Dif : $dif : Min : $min , Max : $max")
    for (i in 1 until nums.size - 1) {
        val cur = nums[i]
        val curMin = Math.min(min, cur - k)
        val curMax = Math.max(max, cur + k)
        val dif1 = max - curMin
        val dif2 = curMax - min
        println("D1 : $dif1 and D2 : $dif2")
        if (dif1 < dif2) {
            min = Math.min(min, curMin)
        } else {
            max = Math.max(max, curMax)
        }
        val minDiff = Math.min(dif1, dif2)
        dif = Math.max(dif, minDiff)
    }

    return dif
}

fun numMusicPlaylists(n: Int, goal: Int, k: Int): Int {
    println(
        numMusicPlaylists(
            2, 3, 0
        )
    )

    return 0
}

fun beautifulSubarrays(nums: IntArray): Long {

    val map = hashMapOf<String, Int>()
    val bitFreqAtLoc = hashMapOf<Int, Int>()
    val initial = java.lang.StringBuilder()
    for (i in 0..31) {
        bitFreqAtLoc[i] = 0
        initial.append(0)
    }
    map[initial.toString()] = 1
    var ans = 0L
    for (n in nums) {
        val binary = Integer.toBinaryString(n)
        val rev = binary.reversed()
        val builder = StringBuilder()
        for (i in 0 until rev.length) {
            if (rev[i] == '1') {
                bitFreqAtLoc[i] = bitFreqAtLoc.getOrDefault(i, 0) + 1
            }
            builder.append(bitFreqAtLoc[i]!! % 2)
        }
        for (i in rev.length..31) {
            builder.append(bitFreqAtLoc[i]!! % 2)
        }
        if (map.containsKey(builder.toString())) {
            ans += map[builder.toString()]!!
        }
        map[builder.toString()] = map.getOrDefault(builder.toString(), 0) + 1
    }

    return ans
}


fun vowelStrings(words: Array<String>, left: Int, right: Int): Int {
    var c = 0
    val vowels = setOf('a', 'e', 'i', 'o', 'u')
    for (i in left..right) {
        val w = words[i]
        if (vowels.contains(w[0]) && vowels.contains(w[w.length - 1])) {
            c++
        }
    }
    return c
}

fun minMalwareSpread(graph: Array<IntArray>, initial: IntArray): Int {

    val adj = Array(graph.size) { mutableSetOf<Int>() }
    for (i in 0 until graph.size) {
        for (j in 0 until graph.size) {
            if (i != j && graph[i][j] == 1) {
                adj[i].add(j)
                adj[j].add(i)
            }
        }
    }
    for (i in 0 until adj.size) {
        println("From : $i to Dest : ${adj[i]}")
    }
    var ans = Int.MAX_VALUE
    var remove = -1
    for (i in 0 until initial.size) {
        val visited = mutableSetOf<Int>()
        var infected = 0
        initial.forEach {
            if (it != initial[it]) infected += dfs(it, adj, visited)
        }
        println("DelNode : ${initial[i]} Infected : $infected")
        if (infected < ans) {
            ans = infected
            remove = initial[i]
        }
    }

    return remove
}

fun dfs(at: Int, adj: Array<MutableSet<Int>>, visited: MutableSet<Int>): Int {
    if (visited.contains(at)) return 0
    var counter = 1
    visited.add(at)
    for (n in adj[at]) {
        if (!visited.contains(n)) {
            counter += dfs(n, adj, visited)
        }
    }
    return counter
}

fun beautifulArray(n: Int): IntArray {

    val nums = mutableListOf<Int>()
    (1..n).forEach {
        nums.add(it)
    }
    println(nums)
    println(
        beautifulArray(
            10
        )
    )

    return intArrayOf()
}

fun distinctSubseqII(s: String): Int {

    println(
        distinctSubseqII(
            "aba"
        )
    )

    return 0
}

fun shortestSuperstring(words: Array<String>): String {
    println(
        shortestSuperstring(
            arrayOf(
                "alex", "loves", "leetcode"
            )
        )
    )
    var target = 0
    for (i in 0 until words.size) {
        target = target or (1 shl i)
    }
    return dpExplore(0, "", words, target)!!
}

fun dpExplore(bit: Int, curStr: String, words: Array<String>, target: Int): String? {
    if (bit == target) return curStr
    var ans: String? = null
    for (i in 0 until words.size) {
        if (bit and (1 shl i) == 0) {
            val newBit = bit or (1 shl i)
            val newString = if (curStr.contains(words[i])) curStr else curStr + words[i]
            val res = dpExplore(newBit, newString, words, target)
            if (ans == null) ans = res
            else if (res != null && res.length < ans.length) {
                ans = res
            }
        }
    }
    return ans
}

fun shareCommonFactor(
    value: Int, factors: HashMap<Int, MutableList<Int>>, adj: MutableSet<String>
) {
    //2,3,5
    val divs = mutableListOf<Int>()
    if (value % 2 == 0) {
        if (factors[2] == null) factors[2] = mutableListOf()
        factors[2]!!.add(value)
        divs.add(2)
    }
    if (value % 3 == 0) {
        if (factors[3] == null) factors[3] = mutableListOf()
        factors[3]!!.add(value)
        divs.add(3)
    }
    if (value % 5 == 0) {
        if (factors[5] == null) factors[5] = mutableListOf()
        factors[5]!!.add(value)
        divs.add(5)
    }
    for (i in 0 until divs.size) {
        for (j in i + 1 until divs.size) {
            val connect = "${divs[i]}|${divs[j]}"
            adj.add(connect)
        }
    }
}


fun minDeletionSize(strs: Array<String>): Int {
    println(
        minDeletionSize(
            arrayOf(
                "ca", "bb", "ac"
            )
        )
    )
    var ans = 0
    for (col in 0 until strs[0].length) {
        var remove = false
        for (row in 1 until strs.size) {
            if (strs[row - 1][col] > strs[row][col]) {
                remove = true
                break
            }
        }
        if (remove) ans++
    }

    return ans
}

fun tallestBillboard(rods: IntArray): Int {
    val memo = hashMapOf<String, Int>()
    val half = rods.sum() / 2
    return dpExplore(0, 0, 0, rods, memo, half)
}

fun dpExplore(first: Int, second: Int, at: Int, rods: IntArray, memo: HashMap<String, Int>, half: Int): Int {
    if (at == rods.size) {
        if (first == second) return first
        return 0
    }
    if (first > half || second > half) return 0
    val key = "$first|$second|$at"
    if (memo.contains(key)) return memo[key]!!
    val giveToFirst = dpExplore(first + rods[at], second, at + 1, rods, memo, half)
    val giveToSecond = dpExplore(first, second + rods[at], at + 1, rods, memo, half)
    val noBody = dpExplore(first, second, at + 1, rods, memo, half)
    val max = Math.max(giveToFirst, Math.max(giveToSecond, noBody))
    memo[key] = max
    return max
}

fun prisonAfterNDays(cells: IntArray, n: Int): IntArray {

    println(
        prisonAfterNDays(
            intArrayOf(0, 1, 0, 1, 1, 0, 0, 1), 7
        )
    )

    return cells
}

fun regionsBySlashes(grid: Array<String>): Int {
    for (i in 0 until grid.size) {
        for (j in 0 until grid[i].length) println(grid[i][j])
    }
    println(
        regionsBySlashes(
            arrayOf(
                "\\/",
                "/\\",
            )
        )
    )

    return 0
}

fun dpExplore(prevCol: Int, atCol: Int, strs: Array<String>, memo: Array<IntArray>): Int {
    if (atCol == strs[0].length) return 0
    if (memo[prevCol + 1][atCol] != -1) return memo[prevCol + 1][atCol]
    var isValid = true
    for (row in 0 until strs.size) {
        if (prevCol != -1) {
            val prevChar = strs[row][prevCol]
            val curChar = strs[row][atCol]
            if (prevChar > curChar) {
                isValid = false
                break
            }
        }
    }
    var dontRemove = Int.MAX_VALUE
    if (isValid) {
        dontRemove = dpExplore(atCol, atCol + 1, strs, memo)
    }
    val remove = 1 + dpExplore(prevCol, atCol + 1, strs, memo)
    val res = Math.min(remove, dontRemove)
    memo[prevCol + 1][atCol] = res
    return res
}

fun minAreaRect(points: Array<IntArray>): Double {
    println(
        minAreaFreeRect(
            arrayOf(
                intArrayOf(0, 0),
                intArrayOf(0, 3),
                intArrayOf(0, 2),
                intArrayOf(2, 1),
                intArrayOf(2, 3),
                intArrayOf(1, 3),
                intArrayOf(3, 2),
                intArrayOf(1, 0)
            )
        )
    )
    val sorted = points.sortedWith(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            if (o1[0] == o2[0]) return o1[1] - o2[1]
            return o1[0] - o2[0]
        }
    })
    val treeMap = TreeMap<Int, MutableSet<Int>>()
    var ans = Int.MAX_VALUE.toDouble()
    for ((x, y) in sorted) {
        if (treeMap.contains(x)) {
            val rightYs = treeMap[x]!!
            for (ys in rightYs) {
                for ((leftX, leftYs) in treeMap) {
                    if (leftYs.contains(y) && leftYs.contains(ys)) {
                        val height = Math.abs(y - ys)
                        val width = Math.abs(x - leftX)
                        ans = Math.min(ans, height * width.toDouble())
                    }
                }
            }
        }
        if (treeMap[x] == null) treeMap[x] = mutableSetOf()
        treeMap[x]!!.add(y)
    }
    return ans
}

fun minAreaFreeRect(points: Array<IntArray>): Double {
    val parallelRect = minAreaRect(points)
    val nonParallelRect = minAreaNonParallelRect(points)
    val min = Math.min(parallelRect, nonParallelRect)
    println("Pal : $parallelRect and NonPar : $nonParallelRect")
    if (min == Int.MAX_VALUE.toDouble()) return 0.0
    return min
}

fun minAreaNonParallelRect(points: Array<IntArray>): Double {
    var min = Int.MAX_VALUE.toDouble()
    val sorted = points.sortedWith(object : Comparator<IntArray> {
        override fun compare(o1: IntArray, o2: IntArray): Int {
            if (o1[0] == o2[0]) return o1[1] - o2[1]
            return o1[0] - o2[0]
        }
    })
    sorted.forEach {
        println(it.toList())
    }
    val treeMap = TreeMap<Int, MutableSet<Int>>()
    for ((x4, y4) in sorted) {
        if (!treeMap.contains(x4)) {
            for ((middleX, middleYs) in treeMap) {
                val distToX4 = x4 - middleX
                val leftMostX = middleX - distToX4
                if (treeMap.contains(leftMostX) && treeMap[leftMostX]!!.contains(y4)) {
                    if (middleYs.size >= 2) {
                        val ysSorted = middleYs.sorted()
                        for (i in 0 until middleYs.size) {
                            for (j in 0 until middleYs.size) {
                                if (i != j) {
                                    val y2 = ysSorted[i]
                                    val y3 = ysSorted[j]
                                    if (Math.abs(y2 - y4) == Math.abs(y3 - y4)) {
                                        val b = distToX4
                                        val a = Math.abs(y2 - y4)
                                        val area = Math.pow(a.toDouble(), 2.0) + Math.pow(b.toDouble(), 2.0)
                                        if (area == 2.0) {
                                            println("X4 :$x4 and Y4 : $y4 with Mid X : $middleX and y2 : $y2 y3: $y3 LeftMostX : $leftMostX")
                                        }
                                        if (area.toInt().toDouble() == area) min = Math.min(min, area)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (treeMap[x4] == null) treeMap[x4] = mutableSetOf()
        treeMap[x4]!!.add(y4)
    }

    return min
}


class MyCalendarThree() {


    fun book(startTime: Int, endTime: Int): Int {
        val obj = MyCalendarThree()

        val st = SegmentTree()
        st.update(ql = 0, qr = 3)
        st.update(ql = 0, qr = 3)
        st.update(ql = 1, qr = 3)
        st.dfs()
        return 0
    }

}

class SegmentTree() {

    val valMap = hashMapOf<Int, Int>()

    fun update(at: Int = 0, l: Int = 0, r: Int = 1000_000_001, ql: Int, qr: Int): Int {
        if (r < ql || qr < l) return 0
        else if (ql <= l && r <= qr) {
            valMap[at] = valMap.getOrDefault(at, 0) + 1
            return valMap[at]!!
        } else {
            val mid = (l + r) / 2
            val left = update(at * 2 + 1, l, mid, ql, qr)
            val right = update(at * 2 + 2, mid + 1, r, ql, qr)
            valMap[at] = Math.max(left, right)
            return valMap[at]!!
        }
    }

    fun dfs(at: Int = 0, l: Int = 0, r: Int = 1000_000_001) {
        if (!valMap.contains(at)) return
        println("Inc : ${valMap[at]} L : $l and R : $r")
        val mid = (l + r) / 2
        dfs(at * 2 + 1, l, mid)
        dfs(at * 2 + 2, mid + 1, r)
    }

}

fun leastOpsExpressTarget(x: Int, target: Int): Int {
    println(
        leastOpsExpressTarget(
            5, 501
        )
    )

    return 0
}


fun pancakeSort(arr: IntArray): List<Int> {
    val swaps = mutableListOf<Int>()
    dfs(arr.size, arr, swaps)
    return swaps
}

fun dfs(at: Int, arr: IntArray, swaps: MutableList<Int>) {
    if (at == 0) return
    if (arr[at - 1] == at) return dfs(at - 1, arr, swaps)
    var pos = -1
    for (i in 0 until at) {
        if (arr[i] == at) {
            pos = i
            break
        }
    }
    swaps.add(pos + 1)
    swap(0, pos, arr)
    swap(0, at - 1, arr)
    swaps.add(at)
    dfs(at - 1, arr, swaps)
}

fun swap(left: Int, right: Int, arr: IntArray) {
    var l = left
    var r = right
    while (l < r) {
        val t = arr[l]
        arr[l] = arr[r]
        arr[r] = t
        l++
        r--
    }
}

fun powerfulIntegers(x: Int, y: Int, bound: Int): List<Int> {

    println(
        powerfulIntegers(
            2, 2, 400000
        )
    )
    val ans = mutableSetOf<Int>()
    dfs(1, 1, x, y, bound, ans)
    return ans.toList()
}

fun dfs(curX: Int, curY: Int, x: Int, y: Int, bound: Int, ans: MutableSet<Int>) {
    if (curX + curY <= bound) {
        ans.add(curX + curY)
    } else return
    dfs(curX * x, curY, x, y, bound, ans)
    dfs(curX, curY * y, x, y, bound, ans)
}

fun isRationalEqual(s: String, t: String): Boolean {

    println(
        isRationalEqual(
            "0.(52)", "0.5(25)"
        )
    )

    return false
}

fun countTriplets(nums: IntArray): Int {

    println(
        countTriplets(
            intArrayOf(
                2, 1, 3
            )
        )
    )

    return 0
}

fun largestNumber(cost: IntArray, target: Int): String {

    val dp = IntArray(target + 1) { Int.MIN_VALUE }
    dp[0] = 0
    for (t in 1..target) {
        for (d in 0..8) {
            if (t >= cost[d]) {
                dp[t] = Math.max(dp[t], dp[t - cost[d]] + 1)
            }
        }
    }
    if (dp[target] < 0) return "0"
    var t = target
    var ans = ""
    while (t > 0) {
        var pos = -1
        for (d in 9 downTo 1) {
            if (t >= cost[d - 1] && (pos == -1 || dp[t - cost[d - 1]] > dp[t - cost[pos - 1]])) {
                pos = d
            }
        }
        ans += "$pos"
        t -= cost[pos - 1]
    }

    return ans
}

fun dpExplore(cost: IntArray, target: Int, memo: Array<String?>): String? {
    if (target == 0) return ""
    if (memo[target] != "0") return memo[target]
    var ans: String? = null
    for (i in 9 downTo 1) {
        if (target - cost[i - 1] >= 0) {
            val res = dpExplore(cost, target - cost[i - 1], memo)
            if (res != null) {
                val concat = "$i" + res
                if (ans == null) ans = concat
                else if (concat.length > ans.length) {
                    ans = concat
                }
            }
        }
    }
    memo[target] = ans
    return ans
}


fun maxTaskAssign(tasks: IntArray, workers: IntArray, pills: Int, strength: Int): Int {
    println(
        maxTaskAssign(
            intArrayOf(5, 9, 8, 5, 9), intArrayOf(1, 6, 4, 2, 6), 1, 5
        )
    )
    val sortedWorkers = workers.sortedWith(Collections.reverseOrder()).toIntArray()
    val queue = PriorityQueue<Int>(Collections.reverseOrder())
    tasks.forEach {
        queue.add(it)
    }
    var life = pills
    var ans = 0
    for (w in sortedWorkers) {
        if (queue.isNotEmpty() && queue.peek() <= w) {
            ans++
            queue.poll()
        } else {
            while (queue.isNotEmpty() && queue.peek() > w + (if (life > 0) strength else 0)) {
                queue.poll()
            }
            if (queue.isNotEmpty() && queue.peek() <= w + (if (life > 0) strength else 0)) {
                ans++
                queue.poll()
            }
            if (life > 0) life--
        }
    }

    return ans
}

fun mctFromLeafValues(A: IntArray): Int {
    var res = 0
    val stack = Stack<Int>()
    stack.push(Int.MAX_VALUE)
    for (a in A) {
        while (stack.peek() <= a) {
            val mid = stack.pop()
            res += mid * stack.peek()
        }
        stack.push(a)
    }
    while (stack.size > 2) {
        res += stack.pop() * stack.peek()
    }
    return res
}

fun mergeStones(stones: IntArray, k: Int): Int {
    if (k > stones.size || (stones.size - k) < k) return -1
    if (!isPossibleToMerge(stones.size, k)) return -1


    return 0
}

fun isPossibleToMerge(size: Int, k: Int): Boolean {
    var cur = size
    while (cur >= k) {
        cur -= k
        cur++
    }
    return cur == 1
}

fun turningOffLamps(row: Int, col: Int, lamps: TreeMap<Int, MutableSet<Int>>, rowsMapping: HashMap<Int, Int>) {
    val neigh = arrayOf(
        intArrayOf(0, 0),
        intArrayOf(-1, -1),
        intArrayOf(-1, 0),
        intArrayOf(-1, 1),
        intArrayOf(0, -1),
        intArrayOf(0, 1),
        intArrayOf(1, -1),
        intArrayOf(1, 0),
        intArrayOf(1, 1)
    )
    for (n in neigh) {
        val r = row + n[0]
        val c = col + n[1]
        val rows = lamps[c] ?: continue
        rows.remove(r)
        rowsMapping[r] = rowsMapping.getOrDefault(r, 0) - 1
        if (rows.isEmpty()) lamps.remove(c)
        if (rowsMapping[r] == 0) rowsMapping.remove(r)
    }
}

fun isIlluminated(
    row: Int, col: Int, lamps: TreeMap<Int, MutableSet<Int>>, rowsMapping: HashMap<Int, Int>
): Boolean {
    var c = col
    if (lamps.containsKey(col)) return true
    //checking to the left
    while (lamps.floorKey(c) != null) {
        val rows = lamps[lamps.floorKey(c)]!!
        val dif = col - lamps.floorKey(c)!!
        val decRow = row - dif
        val incRow = row + dif
        if (rows.contains(decRow) || rows.contains(incRow) || rows.contains(row)) return true
        c = lamps.floorKey(c)
        c--
    }
    //checking to right
    c = col
    while (lamps.ceilingKey(c) != null) {
        val rows = lamps[lamps.ceilingKey(c)]!!
        val diff = lamps.ceilingKey(c)!! - col
        val decRow = row - diff
        val incRow = row + diff
        if (rows.contains(decRow) || rows.contains(incRow) || rows.contains(row)) return true
        c = lamps.ceilingKey(c)
        c++
    }
    return false
}

fun baseNeg2(n: Int): String {
    return convertToBase(n, -2)
}

fun convertToBase(num: Int, base: Int): String {
    var cur = num
    val str = StringBuilder()
    while (cur != 0) {
        var remainder = cur % base
        cur /= base
        if (remainder < 0) {
            remainder += Math.abs(base)
            cur++
        }
        str.append(remainder)
    }
    str.reverse()
    return str.toString()
}

fun minScoreTriangulation(values: IntArray): Int {


    return 0
}

fun lastStoneWeightII(stones: IntArray): Int {


    return 0
}

fun maxEqualRowsAfterFlips(matrix: Array<IntArray>): Int {

    val map = hashMapOf<String, Int>()
    for (mat in matrix) {
        val builder = StringBuilder()
        for (m in mat) {
            builder.append(m)
        }
        map[builder.toString()] = map.getOrDefault(builder.toString(), 0) + 1
    }
    var ans = 0
    for ((key, value) in map) {
        val rev = StringBuilder(key)
        for (i in 0 until rev.length) {
            val bit = rev[i]
            if (bit == '0') rev[i] = '1'
            else rev[i] = '0'
        }
        val res = value + (map[rev.toString()] ?: 0)
        ans = Math.max(ans, res)
    }

    return ans
}

fun addNegabinary(arr1: IntArray, arr2: IntArray): IntArray {
    println(
        addNegabinary(
            intArrayOf(1, 0, 1), intArrayOf(1)
        )
    )
    val answer = StringBuilder()
    var a = arr1.size - 1
    var b = arr2.size - 1
    var carry = 0
    while (a >= 0 || b >= 0) {
        val a1 = if (a >= 0) arr1[a] else 0
        val a2 = if (b >= 0) arr2[b] else 0
        val (res, curCarry) = addTwoBits(a1, a2)
        val (newRes, newCarry) = addTwoBits(res, carry)
        println("Adding $res $carry with new Res $newRes Carry : $newCarry")
        answer.append(newRes)
        carry = newCarry
        a--
        b--
    }
    println(answer.reversed())

    return intArrayOf()
}

//return (res,carry)
fun addTwoBits(a: Int, b: Int): IntArray {
    if (a + b == 2) return intArrayOf(0, -1)
    else if (a + b == -1) {
        return intArrayOf(1, 1)
    } else if (a + b == 1) {
        return intArrayOf(1, 0)
    }
    return intArrayOf(0, 0)
}
