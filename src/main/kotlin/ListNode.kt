import kotlin.random.Random

/*


1 + 1 = 0, carry -1
-1 + 0 = 1, carry 1
1 + 0 = 1, carry 0
0 + 0 = 0, carry 0
0 + 1 = 1, carry 0
-1 + 1 = 0, carry 0



 */
fun main() {


}




fun reservoirSampling(nums: IntArray, k: Int) {
    val ans = IntArray(k) { -1 }
    for (i in 0 until k) {
        ans[i] = nums[i]
    }
    for (i in k until nums.size) {
    }
    println(ans.toList())
}

fun detectCycle(head: ListNode?): ListNode? {

    var slow: ListNode? = head
    var fast: ListNode? = head
    while (slow != null) {
        slow = slow.next
        fast = fast?.next?.next
        if (slow == fast)
            break
    }
    if (slow == null)
        return ListNode(-1)
    var cur = head
    while (cur != slow) {
        cur = cur?.next
        slow = slow?.next
    }

    return cur
}

fun constructListNode(nums: IntArray): ListNode? {
    val dummy = ListNode(-1)
    var cur = dummy
    for (n in nums) {
        val next = ListNode(n)
        cur.next = next
        cur = cur.next!!
    }
    return dummy.next
}

data class ListNode(val `val`: Int) {
    var next: ListNode? = null
}