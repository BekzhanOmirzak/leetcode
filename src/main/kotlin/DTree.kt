/*






 */



fun main() {


}

class DTreeWrapper {
    var root: DTree = DTree()
    var max = 100_001

    fun update(value: Int) {
        updateHelper(root, 0, max, value)
    }

    fun updateHelper(root: DTree, l: Int, r: Int, value: Int) {
        if (l == r && value == l) {
            root.value++
            return
        }
        if (value < l || value > r)
            return
        val mid = (l + r) / 2
        root.value++
        root.extend()
        updateHelper(root.left!!, l, mid, value)
        updateHelper(root.right!!, mid + 1, r, value)
    }

    fun getRangeValue(atLeast: Int): Int {
        return getRangeHelper(root, 0, max, atLeast)
    }

    private fun getRangeHelper(root: DTree?, l: Int, r: Int, max: Int): Int {
        if (root == null) return 0
        if (l == r && max == l)
            return root.value
        else if (l > max)
            return 0
        else if (r <= max)
            return root.value
        val mid = (l + r) / 2
        val left = getRangeHelper(root.left, l, mid, max)
        val right = getRangeHelper(root.right, mid + 1, r, max)
        return left + right
    }

    fun dfs() {
        dfsHelper(root, 0, max)
    }

    private fun dfsHelper(root: DTree?, l: Int, r: Int) {
        if (root == null)
            return
        val mid = (l + r) / 2
        println("Value : ${root.value} ($l,$r)")
        dfsHelper(root.left, l, mid)
        dfsHelper(root.right, mid + 1, r)
    }


}

class DTree(
) {
    var value: Int = 0
    var left: DTree? = null
    var right: DTree? = null

    fun extend() {
        if (left == null) {
            left = DTree()
            right = DTree()
        }
    }

}