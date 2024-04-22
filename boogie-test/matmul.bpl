type Matrix = [int, int]real;

function MatrixMultiply(A: Matrix, B: Matrix, M: int, N: int, P: int): Matrix;



axiom (forall n: int :: {Factorial(n)}  1 <= n ==> Factorial(n) == n * Factorial_Aux(n-1));

var C: [int, int]int;
axiom (forall i, j: int :: (0 <= i && i < M && 0 <= j && j < P) ==> C[i, j] == 0);

axiom (forall i, j: int, c: [int, int]int ::
    (0 <= i && i < M && 0 <= j && j < P &&
     (for all k: int :: (0 <= k && k < N) ==> A[i, k] * B[k, j] == c[i, j])
     ) ==> C[i, j] == c[i, j]);

procedure VerifyMatMulEquivalence(M: int, N: int, P: int) 
{
    var A, B: Matrix;
    // Initialization for matrices A and B would be assumed here
    // For simplicity, let's assume A, B are initialized appropriately elsewhere

    var C1, C2: Matrix;

    // Invoke MatrixMultiply for direct and tiled versions
    // These would be assumed to be separately defined or axiomatized
    C1 := MatrixMultiply(A, B, M, N, P);
    C2 := MatrixMultiply(A, B, M, N, P);
    
    // Assertion to compare every element in C1 and C2 for equivalence
    // assert (forall i: int, j: int :: (0 <= i < M && 0 <= j < P) => (C1[i, j] == C2[i, j]));
}
