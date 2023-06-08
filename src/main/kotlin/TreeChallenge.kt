import com.sun.source.tree.Tree
import java.util.LinkedList
import java.util.PriorityQueue
import java.util.Random
import java.util.TreeMap
import kotlin.math.sign
import kotlin.math.sqrt

/*




study

 */


fun main() {

    val root = getConstructedTreeNode()
    println(treeQueries(root, intArrayOf(2, 3,4)))

}

fun treeQueries(root: TreeNode?, queries: IntArray): IntArray {

    val levelMap = hashMapOf<Int, Int>()
    val deepFreq = TreeMap<Int, Int>()
    val nodeMap = hashMapOf<Int, TreeNode>()
    val removed = mutableSetOf<Int>()
    dfsLevel(root, 1, levelMap, deepFreq, nodeMap)
    val answer = IntArray(queries.size) { 0 }
    for (i in 0 until  queries.size) {
        val value=queries[i]
        val node = nodeMap[value]!!
        val level = levelMap[value]!!
        val temp = mutableListOf<Int>()
        dfs(removed, node, level, deepFreq, temp)
        if (deepFreq.isEmpty())
            answer[i] = 0
        else
            answer[i] = deepFreq.lastKey() - 1
        println("$value     $deepFreq")
        temp.forEach {
            deepFreq[it] = deepFreq.getOrDefault(it, 0) + 1
        }
    }
    println(answer.toList())
    return answer
}

fun dfs(removed: MutableSet<Int>, at: TreeNode?, level: Int, deepFreq: TreeMap<Int, Int>, temp: MutableList<Int>) {
    if (at == null || removed.contains(at.`val`))
        return
    removed.add(at.`val`)
    deepFreq[level] = deepFreq.getOrDefault(level, 0) - 1
    temp.add(level)
    if (deepFreq[level] == 0)
        deepFreq.remove(level)
    dfs(removed, at.left, level + 1, deepFreq, temp)
    dfs(removed, at.right, level + 1, deepFreq, temp)
}

fun dfsLevel(
    at: TreeNode?,
    level: Int,
    levelMap: HashMap<Int, Int>,
    deepFreq: TreeMap<Int, Int>,
    nodeMap: HashMap<Int, TreeNode>
) {
    if (at == null)
        return
    val value = at.`val`
    levelMap[value] = level
    nodeMap[value] = at
    deepFreq[level] = deepFreq.getOrDefault(level, 0) + 1
    dfsLevel(at.left, level + 1, levelMap, deepFreq, nodeMap)
    dfsLevel(at.right, level + 1, levelMap, deepFreq, nodeMap)
}

class Graph(val n: Int, edges: Array<IntArray>) {

    val adj = Array(n) { mutableListOf<Edge>() }

    init {
        for ((from, to, cost) in edges) {
            adj[from].add(Edge(to, cost))
        }
    }

    fun addEdge(edge: IntArray) {
        adj[edge[0]].add(Edge(edge[1], edge[2]))
    }

    fun shortestPath(node1: Int, node2: Int): Int {
        val minDistance = IntArray(n) { Int.MAX_VALUE }
        minDistance[node1] = 0
        //at,cost
        val queue = PriorityQueue(object : Comparator<IntArray> {
            override fun compare(o1: IntArray, o2: IntArray): Int {
                return o1[1] - o2[1]
            }
        })
        queue.add(intArrayOf(node1, 0))
        while (queue.isNotEmpty()) {
            val poll = queue.poll()
            for (next in adj[poll[0]]) {
                val nextCost = poll[1] + next.cost
                if (next.to == node2)
                    return nextCost
                if (nextCost < minDistance[next.to]) {
                    minDistance[next.to] = nextCost
                    queue.add(intArrayOf(next.to, nextCost))
                }
            }
        }
        return -1
    }

    data class Edge(val to: Int, val cost: Int)

}

fun replaceValueInTree(root: TreeNode?): TreeNode? {
    val map = hashMapOf<TreeNode, Int>()
    saveChildrenSum(root, map)
    val queue = LinkedList<ParentNode>()
    queue.add(ParentNode(root, null))
    var totalSum = 0
    while (queue.isNotEmpty()) {
        val size = queue.size
        var curSum = 0
        for (i in 0 until size) {
            val pop = queue.pop()
            if (pop.parent == null) {
                pop.cur?.`val` = 0
            } else {
                pop.cur?.`val` = totalSum - (map[pop.parent]!!)
            }
            if (pop.cur?.left != null) {
                curSum += pop.cur.left!!.`val`
                queue.add(ParentNode(pop.cur.left, pop.cur))
            }
            if (pop.cur?.right != null) {
                curSum += pop.cur.right!!.`val`
                queue.add(ParentNode(pop.cur.right, pop.cur))
            }
        }
        totalSum = curSum
    }
    return root
}

data class ParentNode(val cur: TreeNode?, val parent: TreeNode?)

fun saveChildrenSum(root: TreeNode?, map: HashMap<TreeNode, Int>) {
    if (root == null) return
    var sum = root.left?.`val` ?: 0
    sum += root.right?.`val` ?: 0
    map[root] = sum
    saveChildrenSum(root.left, map)
    saveChildrenSum(root.right, map)
}

var root: TreeNode? = null
fun deleteNode(root: TreeNode?, key: Int): TreeNode? {
    val res = deleteNode(getConstructedTreeNode(), 3)
    dfs(res)
    return root
}

class TreeAncestor(n: Int, parent: IntArray) {

    val ancestors = Array(n) { IntArray(20) { -1 } }

    init {
        for (i in 1 until parent.size) {
            ancestors[i][0] = parent[i]
        }
        for (i in 1..19) {
            for (j in 1 until parent.size) {
                val prevParent = ancestors[j][i - 1]
                if (prevParent != -1)
                    ancestors[j][i] = ancestors[prevParent][i - 1]
            }
        }
    }

    fun getKthAncestor(node: Int, k: Int): Int {
        val binary = Integer.toBinaryString(k).reversed()
        var atNode = node
        for (bit in 0 until binary.length) {
            if (binary[bit] == '1') {
                atNode = ancestors[atNode][bit]
                if (atNode == -1)
                    return -1
            }
        }
        return atNode
    }

}


fun longestUnivaluePath(root: TreeNode?): Int {
    dfsUniValue(root)
    return longestUniValue
}

var longestUniValue = 0

fun dfsUniValue(root: TreeNode?): Int {
    if (root == null) return 0
    var left = dfsUniValue(root.left)
    var right = dfsUniValue(root.right)
    if (root.`val` == root.left?.`val`) {
        left++
    }
    if (root.`val` == root.right?.`val`) {
        right++
    }
    longestUniValue = Math.max(longestUniValue, left + right)
    return Math.max(left, right)
}

fun testing(): Int {
    var first = 0
    var second = 0
    for (i in 0 until 20) {
        val cur = Random().nextInt(20)
        println("Cur : $cur")
        if (cur > first) {
            first = cur
        } else if (cur > second) {
            second = cur
        }
    }
    println("First : $first and Second : $second")

    return 0
}

fun longestPathWithDiffCharacters(board: IntArray): Int {
    val chars = charArrayOf('a', 'b', 'd', 'k', 'a')

    return 0
}

fun isCompleteTree(root: TreeNode?): Boolean {
    if (root == null) return true
    val queue = LinkedList<TreeNode>()
    queue.add(root)
    var power = 1
    while (queue.isNotEmpty()) {
        val size = queue.size
        var c = 0
        power *= 2
        var facedNull = false
        for (i in 0 until size) {
            val poll = queue.poll()
            if (poll.left != null) {
                if (facedNull)
                    return false
                c++
                queue.add(poll.left!!)
            } else {
                facedNull = true
            }
            if (poll.right != null) {
                if (facedNull)
                    return false
                c++
                queue.add(poll.right!!)
            } else {
                facedNull = true
            }
        }
        if (c != power) {
            while (queue.isNotEmpty()) {
                val poll = queue.poll()
                if (poll.left != null || poll.right != null)
                    return false
            }
        }
    }
    return true
}

fun depthTree(root: TreeNode?): Int {
    if (root == null) return 0
    return 1 + Math.max(depthTree(root.left), depthTree(root.right))
}

var preIndex = 0
var postIndex = 0

fun constructFromPrePost(preorder: IntArray, postorder: IntArray): TreeNode? {
    val root = TreeNode(preorder[preIndex++])
    if (root.`val` != postorder[postIndex]) {
        root.left = constructFromPrePost(preorder, postorder)
    }
    if (root.`val` != postorder[postIndex]) {
        root.right = constructFromPrePost(preorder, postorder)
    }
    postIndex++
    return null
}

fun buildTree(preOrder: LinkedList<Int>, postList: MutableList<Int>): TreeNode? {
    if (postList.isEmpty())
        return null
    if (postList.size == 1)
        return TreeNode(preOrder.poll())
    val pop = preOrder.poll()
    val root = TreeNode(pop)
    val peek = preOrder.peek()
    root.left = buildTree(preOrder, postList.subList(0, postList.indexOf(peek) + 1))
    root.right = buildTree(preOrder, postList.subList(postList.indexOf(peek) + 1, postList.size - 1))
    return root
}

fun flipEquiv(root1: TreeNode?, root2: TreeNode?): Boolean {
    if (root1 == null && root2 == null)
        return true
    if (root1 == null || root2 == null)
        return false
    if (root1.`val` != root2.`val`)
        return false
    return (flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)) || (flipEquiv(
        root1.left,
        root2.right
    ) && flipEquiv(root1.right, root2.left))
}

fun distributeCoins(root: TreeNode?): Int {
    dfs(root)
    return moves
}

var moves = 0

fun dfs(root: TreeNode?): Move {
    if (root == null) return Move(false, 0)
    if (root.left == null && root.right == null) {
        if (root.`val` == 0) {
            moves++
            return Move(false, 1)
        }
        moves += (root.`val` - 1)
        return Move(true, root.`val` - 1)
    }
    val left: Move = dfs(root.left)
    val right: Move = dfs(root.right)
    val coinMove = IntArray(2) { 0 }
    if (left.isCoin)
        coinMove[0] += left.count
    else
        coinMove[1] += left.count
    if (right.isCoin)
        coinMove[0] += right.count
    else
        coinMove[1] += right.count
    if (root.`val` == 0) {
        if (coinMove[0] != 0)
            coinMove[0]--
        else
            coinMove[1]++
    } else {
        coinMove[0] += (root.`val` - 1)
    }
    val min = Math.min(coinMove[0], coinMove[1])
    coinMove[0] -= min
    coinMove[1] -= min
    moves += coinMove[0]
    moves += coinMove[1]
    if (coinMove[0] != 0)
        return Move(true, coinMove[0])
    return Move(false, coinMove[1])
}

data class Move(val isCoin: Boolean, var count: Int)

fun insertIntoMaxTree(root: TreeNode?, `val`: Int): TreeNode? {
    if (root == null)
        return TreeNode(`val`)
    if (root.`val` < `val`) {
        val newRoot = TreeNode(`val`)
        newRoot.left = root
        return newRoot
    }
    val res = insertIntoMaxTree(root.right, `val`)
    root.right = res
    return root
}

fun numSquarefulPerms(nums: IntArray): Int {

    val freq = hashMapOf<Int, Int>()
    nums.forEach {
        freq[it] = freq.getOrDefault(it, 0) + 1
    }
    var c = 0
    val keys = freq.keys
    for (k in keys) {
        freq[k] = freq.getOrDefault(k, 0) - 1
        c += dfs(1, k, freq, nums.size)
        freq[k] = freq.getOrDefault(k, 0) + 1
    }
    return c
}

fun dfs(at: Int, prev: Int, freq: HashMap<Int, Int>, len: Int): Int {
    if (at >= len)
        return 1
    val keys = freq.keys
    var c = 0
    for (k in keys) {
        if (freq[k]!! > 0 && isPerfectSquare(k + prev)) {
            freq[k] = freq.getOrDefault(k, 0) - 1
            c += dfs(at + 1, k, freq, len)
            freq[k] = freq.getOrDefault(k, 0) + 1
        }
    }
    return c
}

fun isPerfectSquare(num: Int): Boolean {
    if (num == 1 || num == 0) return true
    val sqrt = sqrt(num.toDouble())
    if (sqrt <= 1) return false
    return sqrt.toInt() * sqrt.toInt() == num
}


val cached = mutableSetOf<MutableList<Int>>()
fun dpExplore(pos: Int, nums: IntArray, usedBit: Int, prev: Int, ans: MutableList<Int>): Int {
    if (pos >= nums.size) {
        if (cached.contains(ans))
            return 0
        cached.add(ans)
        return 1
    }
    var c = 0
    for (i in 0 until nums.size) {
        if (usedBit and (1 shl i) == 0 && isPerfectSquare(prev + nums[i])) {
            val newBit = usedBit or (1 shl i)
            ans[pos] = nums[i]
            c += dpExplore(pos + 1, nums, newBit, nums[i], ans)
        }
    }
    return c
}

fun constructMaximumBinaryTree(arr: IntArray): TreeNode? {
    if (arr.isEmpty()) return null
    var index = -1
    var max = Integer.MIN_VALUE
    for (i in 0 until arr.size) {
        if (arr[i] > max) {
            max = arr[i]
            index = i
        }
    }
    val head = TreeNode(max)
    val left = IntArray(index) { 0 }
    for (i in 0 until index) {
        left[i] = arr[i]
    }
    val right = IntArray(arr.size - index - 1) { 0 }
    var j = 0
    for (i in index + 1 until arr.size) {
        right[j] = arr[i]
        j++
    }
    head.left = constructMaximumBinaryTree(left)
    head.right = constructMaximumBinaryTree(right)
    return head


}

fun inOrderTraversal(root: TreeNode?, list: MutableList<Int>) {
    if (root == null) return
    inOrderTraversal(root.left, list)
    list.add(root.`val`)
    inOrderTraversal(root.right, list)
}


class TreeNode(var `val`: Int) {
    var left: TreeNode? = null
    var right: TreeNode? = null
}


fun getConstructedTreeNode(): TreeNode {
    val tr1 = TreeNode(1)
    val tr2 = TreeNode(2)
    val tr3 = TreeNode(3)
    val tr4 = TreeNode(4)
    val tr5 = TreeNode(5)
    val tr6 = TreeNode(6)
    val tr7 = TreeNode(7)
    tr1.left = tr2
    tr1.right = tr3
    tr2.left = tr4
    tr2.right = tr5
    tr3.left = tr6
    tr3.right = tr7
    return tr1
}

fun dfsTreeNode(root: TreeNode?) {
    if (root == null) return
    println(root.`val`)
    dfsTreeNode(root.left)
    dfsTreeNode(root.right)
}
