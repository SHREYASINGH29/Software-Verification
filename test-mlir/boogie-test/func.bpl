// the below code is in boogie for verifying matrix multiplication

const m: int;
const n: int;
const p: int;

axiom (m > 0 && n > 0 && p > 0);

function A(int, int) : int;
function B(int, int) : int;
function R1(int, int) : int;
function R2(int, int) : int;

procedure MatMul(){
    var i, j, k: int;
    var temp: [int, int]int;

    // Initialize matrix C to zeros
    i := 0;
    while (i < m)
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
    	invariant 0 <= j && j <= p;
	invariant temp[i, j] == 1;
        {
            temp[i, j] := 1;
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
                temp[i, j] := temp[i, j] + (A(i, k) * B(k, j));
                k := k + 1;
            }
            j := j + 1;
        }
        i := i + 1;
    }

    // Assign the computed values back to C
    i := 0;
    while (i < m)
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
    	invariant 0 <= j && j <= p;
        {
            assume R1(i, j) == temp[i, j];
            j := j + 1;
        }
        i := i + 1;
    }
}


procedure MatMul1()
{
    var i, j, k: int;
    var temp: [int, int]int;

    // Initialize matrix C to zeros
    i := 0;
    while (i < m)
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
    	invariant 0 <= j && j <= p;
        {
            temp[i, j] := 1;
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
                temp[i, j] := temp[i, j] + (A(i, k) * B(k, j));
                k := k + 1;
            }
            j := j + 1;
        }
        i := i + 1;
    }

    // Assign the computed values back to C
    i := 0;
    while (i < m)
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < p)
    	invariant 0 <= j && j <= p;
        {
            assume R2(i, j) == temp[i, j];
            j := j + 1;
        }
        i := i + 1;
    }
}

procedure CheckEquivalence(){
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
        {
            assert R1(i, j) == R2(i, j);
            j := j + 1;
        }
        i := i + 1;
    }
}

// procedure CheckEquivalence()
// {
//     var C1: [int, int]int;
//     var C2: [int, int]int;
//     var i, j: int;
//
//     call MatMul();
//     i := 0;
//     while (i < m) {
//         j := 0;
//         while (j < p) {
//             C1[i, j] := R1(i, j);
//             j := j + 1;
//         }
//         i := i + 1;
//     }
//
//     call MatMul1();
//     i := 0;
//     while (i < m) {
//         j := 0;
//         while (j < p) {
//             C2[i, j] := R2(i, j);
//             j := j + 1;
//         }
//         i := i + 1;
//     }
//
//     // Compare C1 and C2
//     i := 0;
//     while (i < m)
//     {
//         j := 0;
//         while (j < p)
//         {
//             // assert C1[i, j] == C2[i, j];  // Check for equivalence
//             assert R1(i, j) == R2(i, j);  // Check for equivalence
//             j := j + 1;
//         }
//         i := i + 1;
//     }
// }

// procedure CheckEquivalence()
// {
//     var C1: [int, int]int;
//     var C2: [int, int]int;
//     var i, j, k: int;
//
//     call MatMul();
//     // Assume after calling MatMul, the state of C is stored to C1
//     i := 0;
//     while (i < m)
//     {
//         j := 0;
//         while (j < p)
//         {
//             C1[i, j] := C(i, j);
//             j := j + 1;
//         }
//         i := i + 1;
//     }
//
//     call MatMul1();
//     // Assume after calling matmul1, the state of C is stored to C2
//     i := 0;
//     while (i < m)
//     {
//         j := 0;
//         while (j < p)
//         {
//             C2[i, j] := C(i, j);
//             j := j + 1;
//         }
//         i := i + 1;
//     }
//
//     // Compare C1 and C2
//     i := 0;
//     while (i < m)
//     {
//         j := 0;
//         while (j < p)
//         {
//             assert C1[i, j] == C2[i, j]; // Check for equivalence
//
//             invariant forall k1, k2 :: (0 <= k1 < i && 0 <= k2 < p) ==> (C1[k1, k2] == C2[k1, k2]);
//             invariant 0 <= j <= p;
//             j := j + 1;
//         }
//         invariant forall k1, k2 :: (0 <= k1 < i && 0 <= k2 < p) ==> (C1[k1, k2] == C2[k1, k2]);
//         invariant 0 <= i <= m;
//         i := i + 1;
//     }
// }

// procedure CheckEquivalence2()
// {
//     var C1: [int, int]int;
//     var C2: [int, int]int;
//     var i, j, k: int;
//
//     call MatMul();
//     // Store the state of C to C1
//     i := 0;
//     while (i < m)
//     {
//         j := 0;
//         while (j < p)
//         {
//             C1[i, j] := C(i, j);
//             j := j + 1;
//         }
//         i := i + 1;
//     }
//
//     call matmul1();
//     // Store the state of C to C2
//     i := 0;
//     while (i < m)
//     {
//         j := 0;
//         while (j < p)
//         {
//             C2[i, j] := C(i, j);
//             j := j + 1;
//         }
//         i := i + 1;
//     }
//
//     // Compare C1 and C2
//     i := 0;
//     while (i < m)
//     {
//         invariant (forall k1, k2 :: 0 <= k1 && k1 < i && 0 <= k2 && k2 < p ==> C1[k1, k2] == C2[k1, k2]);
//         invariant (0 <= i && i <= m);
//
//         j := 0;
//         while (j < p)
//         {
//             invariant (forall k1, k2 :: 0 <= k1 && k1 <= i && 0 <= k2 && k2 < j ==> C1[k1, k2] == C2[k1, k2]);
//             invariant (0 <= j && j <= p);
//
//             assert C1[i, j] == C2[i, j];  // Check for equivalence
//             j := j + 1;
//         }
//         i := i + 1;
//     }
// }
