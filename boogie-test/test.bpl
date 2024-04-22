// write boogie code to test equivalence of two matmul implementations

type Matrix = [int, int]real;

// define a function to multiply two matrices 
function MatrixMultiply(A: Matrix, B:Matrix, M:int, N:int, P:int): Matrix;

axiom (forall A: Matrix, B: Matrix, C: Matrix, M: int, N: int, P: int, i: int, j: int ::
      (0 <= i && i < M && 0 <= j && j < P) ==>
      C[i, j] == sum(k: int :: (0 <= k && k < N)(A[i, k] * B[k, j])));

// axiom (forall A: Matrix, B: Matrix, C: Matrix, M: int, N: int, P: int, i: int, j: int ::
//       (0 <= i && i < M && 0 <= j && j < P) ==>
//       (exists k: int :: (0 <= k && k < N && C[i, j] == A[i, k] * B[k, j])));

procedure VerifyMatMul(A: Matrix, B: Matrix, M: int, N: int, P: int)
{
  var C1, C2: Matrix;
  C1 := MatrixMultiply(A, B, M, N, P);
  C2 := MatrixMultiply(A, B, M, N, P);
  assert (forall i: int, j: int :: (0 <= i && i < M && 0 <= j && j < P) ==> C1[i, j] == C2[i, j]);
}

