const m: int;
const n: int;
const l: int;

axiom m > 0 && n > 0 && l > 0;

procedure simpleMatmul()
{
    var count1, count2, i, j: int;
    count1 := 0;
    count2 := 0;
    
    
    i := 0;
    while (i < m) 
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < n)
        invariant 0 <= j && j <= n;
        {
            count1 := count1 + 1;
            j := j + 1;
        }
        i := i + 1;
    }
    
    i := 0;
    while (i < m)
    invariant 0 <= i && i <= m;
    {
        j := 0;
        while (j < n) 
        invariant 0 <= j && j <= n;
        {
            count2 := count2 + 1;
            j := j + 1;
        }
        i := i + 1;
    }
    
    assert(count1 == count2); // This should hold and Boogie should verify it.
}

procedure main()
{
    call simpleMatmul();
}
