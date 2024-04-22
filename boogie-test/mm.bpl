method Main()
{
	var m1: array2<int>, m2: array2<int>, m3: array2<int>;
	m1 := new int[2,3];
	m2 := new int[3,1];
	m1[0,0] := 1; m1[0,1] := 2; m1[0,2] := 3;
	m1[1,0] := 4; m1[1,1] := 5; m1[1,2] := 6;
	m2[0,0] := 7;
	m2[1,0] := 8;
	m2[2,0] := 9;
	m3 := Multiply'(m1, m2);
	PrintMatrix(m1);
	print "\n*\n";
	PrintMatrix(m2);
	print "\n=\n";
	PrintMatrix(m3);
}
 
method PrintMatrix(m: array2<int>)
	requires m != null
{
	var i: nat := 0;
	while (i < m.Length0)
	{
		var j: nat := 0;
		print "\n";
		while (j < m.Length1)
		{
			print m[i,j];
			print "\t";
			j := j + 1;
		} 
		i := i + 1;
	} 
	print "\n";
}
predicate MM(m1: array2<int>, m2: array2<int>, m3: array2<int>)
{ // m3 is the result of multiplying the matrix m1 by the matrix m2
	m1 != null && m2 != null && m3 != null &&
	m1.Length1 == m2.Length0 && m3.Length0 == m1.Length0 && m3.Length1 == m2.Length1 &&
	forall i,j :: 0 <= i < m3.Length0 && 0 <= j < m3.Length1 ==> m3[i,j] == RowColumnProduct(m1,m2,i,j)
}
 
function RowColumnProduct(m1: array2<int>, m2: array2<int>, row: nat, column: nat): int
	requires m1 != null && m2 != null && m1.Length1 == m2.Length0
	requires row < m1.Length0 && column < m2.Length1 
{
	RowColumnProductFrom(m1, m2, row, column, 0)
}
 
function RowColumnProductFrom(m1: array2<int>, m2: array2<int>, row: nat, column: nat, k: nat): int
	requires m1 != null && m2 != null && k <= m1.Length1 == m2.Length0
	requires row < m1.Length0 && column < m2.Length1
	decreases m1.Length1 - k
{
	if k == m1.Length1 then 0 else m1[row,k]*m2[k,column] + RowColumnProductFrom(m1, m2, row, column, k+1)
}
 
function RowColumnProductTo(m1: array2<int>, m2: array2<int>, row: nat, column: nat, k: nat,i:nat): int
	requires m1 != null && m2 != null && k <= m1.Length1 == m2.Length0
	requires row < m1.Length0 && column < m2.Length1 && i < m1.Length1 == m2.Length0
  requires k<=i
	decreases i - k
{
	if k == i then 0 else m1[row,k]*m2[k,column] + RowColumnProductTo(m1, m2, row, column, k+1,i)
}
 
predicate MMROW(m1: array2<int>, m2: array2<int>, m3: array2<int>,row:nat,col:nat)
{ // m3 is the result of multiplying the matrix m1 by the matrix m2
	m1 != null && m2 != null && m3 != null &&
	m1.Length1 == m2.Length0 && m3.Length0 == m1.Length0 && m3.Length1 == m2.Length1 &&
  row <= m1.Length0 && col <= m2.Length1 &&
	forall i,j :: 0 <= i < row && 0 <= j < col ==> m3[i,j] == RowColumnProduct(m1,m2,i,j)
 
}
 
predicate MMCOL(m1: array2<int>, m2: array2<int>, m3: array2<int>,row:nat,col:nat)
{ // m3 is the result of multiplying the matrix m1 by the matrix m2
	m1 != null && m2 != null && m3 != null &&
	m1.Length1 == m2.Length0 && m3.Length0 == m1.Length0 && m3.Length1 == m2.Length1 &&
  row <= m1.Length0 && col <= m2.Length1 &&
	forall i,j :: 0 <= i < row && 0 <= j < col ==> m3[i,j] == RowColumnProduct(m1,m2,i,j)
 
}
predicate MMI(m1: array2<int>, m2: array2<int>, m3: array2<int>,row:nat,col:nat,i:nat)
{ // m3 is the result of multiplying the matrix m1 by the matrix m2
	m1 != null && m2 != null && m3 != null &&
	m1.Length1 == m2.Length0 && m3.Length0 == m1.Length0 && m3.Length1 == m2.Length1 &&
  row < m1.Length0 && col < m2.Length1 && 0<=i<m1.Length1 &&
	forall n,j :: 0 <= n < row && 0 <= j < col ==> m3[n,j] == RowColumnProduct(m1,m2,n,j)
  && m3[row,col] == RowColumnProductTo(m1, m2, row, col ,0,i)
}
 
method Multiply'(m1: array2<int>, m2: array2<int>) returns (m3: array2<int>)
	requires m1 != null && m2 != null
  requires m1.Length1 > 0 && m2.Length0 > 0
	requires m1.Length1 == m2.Length0
	ensures MM(m1, m2, m3)
{
	m3 := new int[m1.Length0, m2.Length1];
  var row:nat := 0;
  var col:nat  := 0;
  var i:nat  := 0;
 
  while(row != m1.Length0)
    invariant MMROW(m1, m2, m3,row, col)
    invariant (0<=row<= m1.Length0)
		decreases m1.Length0 - row
  {
      while(col != m2.Length1)
      invariant MMCOL(m1, m2, m3,row, col)
      invariant (0<=col<= m2.Length1)
  		decreases m2.Length1 - col
      {
          while(i != m1.Length1)
            invariant MMI(m1, m2, m3,row, col,i)
            invariant (i<= m1.Length1==m2.Length0)&&(0<=col<= m2.Length1)&&(0<=row<= m1.Length0)
    		    decreases m1.Length1 - i
           {   
             m3[row,col]:= m3[row,col]+(m1[row,i]*m2[i,col]);
             i := i+1;
           }
           col := col+1;
           i := 0;
      }
      row := row+1;
      col:= 0;
  }
}
