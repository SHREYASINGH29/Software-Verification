const m: int;
const n: int;
const p: int;

axiom (m > 0 && n > 0 && p > 0);

function A(int, int) : int;
function B(int, int) : int;

var R1: [int]int;
var R2: [int]int;

procedure MatMul() modifies R1;
{
    var i, j, k: int;
    var temp: [int]int;

    // Initialize matrix temp to zeros
    i := 0;
    while (i < m)
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
        invariant 0 <= j && j <= p;
        {
            temp[i*p+j] := 0;
            j := j + 1;
        }
        i := i + 1;
    }

    // Perform matrix multiplication
    i := 0;
    while (i < m)
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
        invariant 0 <= j && j <= p;
        {
            k := 0;
            while (k < n)
            invariant 0 <= k && k <= n;
            {
                temp[i*p+j] := temp[i*p+j] + (A(i, k) * B(k, j));
                k := k + 1;
            }
            j := j + 1;
        }
        i := i + 1;
    }

    // Assign the computed values back to R1
i := 0;
   while (i < m)
	   invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
        invariant 0 <= j && j <= p;
        {
            R1[i*p+j] := temp[i*p+j];
            j := j + 1;
        }
        i := i + 1;
    }
}

procedure MatMul1()
    modifies R2;
{
    var i, j, k: int;
    var temp: [int]int;

    // Initialize matrix temp to zeros
    i := 0;
    while (i < m)
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
        invariant 0 <= j && j <= p;
        {
            temp[i*p+j] := 0;
            j := j + 1;
        }
        i := i + 1;
    }

    // Perform matrix multiplication
    i := 0;
    while (i < m)
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
        invariant 0 <= j && j <= p;
        {
            k := 0;
            while (k < n)
            invariant 0 <= k && k <= n;
            {
                temp[i*p+j] := temp[i*p+j] + (A(i, k) * B(k, j));
                k := k + 1;
            }
            j := j + 1;
        }
        i := i + 1;
    }

    // Assign the computed values back to R2
    i := 0;
    while (i < m)
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
        invariant 0 <= j && j <= p;
        {
            R2[i*p+j] := temp[i*p+j];
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
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
        invariant 0 <= j && j <= p;
	// invariant R1[i*p+j] == R2[i*p+j];
        {
            assert R1[i*p+j] == R2[i*p+j];
            j := j + 1;
        }
        i := i + 1;
    }
}
