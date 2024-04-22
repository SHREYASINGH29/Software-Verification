var A: [int]int;
var B: [int]int;
var R1: [int]int; // Result from RowWiseMatMul
var R2: [int]int; // Result from ColumnWiseMatMul

const N: int;

axiom N > 0;

function MatIndex(i: int, j: int): int
{
    N * i + j
}

procedure RowWiseMatMul() modifies R1;
{
    var i, j, k, sum: int;
    i := 0;
    while (i < N) 
    {
        j := 0;
        while (j < N) 
        {
            sum := 0;
            k := 0;
            while (k < N) 
            {
                sum := sum + A[MatIndex(i, k)] * B[MatIndex(k, j)];
                k := k + 1;
            }
            R1[MatIndex(i, j)] := sum;
            j := j + 1;
        }
        i := i + 1;
    }
}

procedure ColumnWiseMatMul() modifies R2;
{
    var i, j, k, sum: int;
    j := 0;
    while (j < N) 
    {
        i := 0;
        while (i < N) 
        {
            sum := 0;
            k := 0;
            while (k < N) 
            {
                sum := sum + A[MatIndex(i, k)] * B[MatIndex(k, j)];
                k := k + 1;
            }
            R2[MatIndex(i, j)] := sum;
            i := i + 1;
        }
        j := j + 1;
    }
}

procedure InitializeMatrices() modifies A, B;
{
    var i, j: int;

    // Initialize A and B for a simple case, e.g., identity matrices or a deterministic pattern
    i := 0;
    while (i < N) {
        j := 0;
        while (j < N) {
            if (i == j) {
                A[MatIndex(i,j)] := 1; // Simplified pattern, e.g., identity matrix
                B[MatIndex(i,j)] := 1;
            } else {
                A[MatIndex(i,j)] := 0;
                B[MatIndex(i,j)] := 0;
            }
            j := j + 1;
        }
        i := i + 1;
    }
}

procedure CheckEquivalence() modifies R1, R2, A, B;
{
    var i, j: int;
    call InitializeMatrices();
    call RowWiseMatMul();
    call ColumnWiseMatMul();

    i := 0;
    while (i < N) {
        j := 0;
        while (j < N) {
            assert R1[MatIndex(i, j)] == R2[MatIndex(i, j)]; // Verifying equivalence
            j := j + 1;
        }
        i := i + 1;
    }
}
