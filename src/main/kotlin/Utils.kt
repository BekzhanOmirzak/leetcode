fun convertToIntArray(str: String): Array<IntArray> {
    return emptyArray()
}


fun getSumOfNums(n: Int): Int {
    return (n * (n + 1)) / 2
}

fun Array<String>.toCharArray(): CharArray {
    return this.map { it[0] }.toCharArray()
}

fun parseToArrayOfIntArray(str: String): Array<IntArray> {
    val ans = mutableListOf<IntArray>()
    val trim = str.trim()
    var at = 1
    while (at < str.length - 1) {
        if (str[at] == '[') {
            at++
            val builder=StringBuilder()
            while(at<str.length && str[at]!=']'){
                builder.append(str[at++])
            }
            val split=builder.toString().split(',').map { it.toInt() }.toIntArray()
            ans.add(split)
            at+=2
            builder.clear()
        }
    }
    return ans.toTypedArray()
}