



fun main() {


    val (city, people) = readLine().toString().split(" ").map { it.toLong() }
    val minCap = readLine().toString().split(" ").map { it.toInt() }.min()!!
    val times = readLine().toString().split(" ").map { it.toInt() }
    var total = 0L
    for (t in times) {
        total += t
    }
    var counter=people/minCap
    if(people%minCap==0L)
        counter--
    println(counter+total)
}

