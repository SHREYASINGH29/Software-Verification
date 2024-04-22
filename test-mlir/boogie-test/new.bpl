var A: [int, int]int;
var B: [int, int]int;
var R1: [int, int]int;
var R2: [int, int]int;

const m: int;
const n: int;
const p: int;

axiom m > 0 && n > 0 && p > 0;

procedure MatMul() modifies R1;
{
    var i, j, k, sum: int;
    i := 0;
    while (i < m)
    invariant i <= m;
    {
        j := 0;
        while (j < p)
        invariant j <= p;
        {
            sum := 0;
            k := 0;
            while (k < n)
            invariant k <= n;
            {
                sum := sum + A[i, k] * B[k, j];
                k := k + 1;
            }
            R1[i, j] := sum;
            j := j + 1;
        }
        i := i + 1;
    }
}

procedure MatMul1() modifies R2;
{
    var i, j, k, sum: int;
    i := 0;
    while (i < m)
    invariant i <= m;
    {
        j := 0;
        while (j < p)
        invariant j <= p;
        {
            sum := 0;
            k := 0;
            while (k < n)
            invariant k <= n;
            {
                sum := sum + A[i, k] * B[k, j];
                k := k + 1;
            }
            R2[i, j] := sum;
            j := j + 1;
        }
        i := i + 1;
    }
}

procedure CheckEquivalence() modifies R1, R2;
{
    var i, j: int;
    call MatMul();
    call MatMul1();

    i := 0;
    while (i < m)
    invariant i <= m;
    {
        j := 0;
        while (j < p)
        invariant j <= p;
        {
            assert R1[i, j] == R2[i, j];
            j := j + 1;
        }
        i := i + 1;
    }
}
