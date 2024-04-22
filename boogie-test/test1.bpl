type Matrix = [int, int]int;

function IncrementRowWise(A: Matrix, rows: int, cols: int): Matrix
{
    [i: int, j: int] 
        if (0 <= i && i < rows && 0 <= j && j < cols) 
        then A[i,j] + 1 
        else A[i,j]
}

function IncrementColWise(A: Matrix, rows: int, cols: int): Matrix
{
    [i: int, j: int] 
        if (0 <= i && i < rows && 0 <= j && j < cols) 
        then A[i,j] + 1 
        else A[i,j]
}

procedure VerifyIncrementEquivalence(A: Matrix, rows: int, cols: int)
{
    assert (forall i: int, j: int :: 
        (0 <= i && i < rows && 0 <= j && j < cols) =>
        IncrementRowWise(A, rows, cols)[i,j] == IncrementColWise(A, rows, cols)[i,j]);
}
