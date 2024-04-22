procedure verifyCommutativity(a: int, b: int)
{
    var result1: int;
    var result2: int;
    result1 := a * b;
    result2 := b * a;
    assert result1 == result2; // Verifying commutativity
}

procedure verifyDistributivity(a: int, b: int, c: int)
{
    var leftSide: int;
    var rightSide: int;
    leftSide := a * (b + c);
    rightSide := a * b + a * c;
    assert leftSide == rightSide; // Verifying distributivity over addition
}
