procedure checkEquality(m: real, T: real)
    requires T != 0.0; // To avoid division by zero
{
    var l1: real;
    var i1: real;
    var i2: real;

    l1 := m / T;
    i1 := l1 * T;
    i2 := m;

    assert i1 == i2;
}

procedure check(m: real, n: real, l: real) 
    requires m > 0.0 && n > 0.0 && l > 0.0;
{
    var a1, a2, b1, b2, c1, c2: real;
    a1 := n;
    a2 := m;

    b1 := m;
    b2 := l;

    c1 := l;
    c2 := n;

    assert(a1 == a2 || a1 == b2 || a1 == c2);
    assert(b1 == a2 || b1 == b2 || b1 == c2);
    assert(c1 == a2 || c1 == b2 || c1 == c2);
    assert(a1 * b1 * c1 == a2 * b2 * c2);
}

procedure CheckRangesCoverTheSame()
{
    var Data1StartIdx: int;
    var Data1Size: int;
    var Data2StartIndices: [int]int;
    var Data2Sizes: [int]int;

    var N2: int;

    N2 := 2;
    Data1StartIdx := 0; Data1Size := 5;

    Data2StartIndices[0] := 0; Data2Sizes[0] := 3;
    Data2StartIndices[1] := 3; Data2Sizes[1] := 2;

    // 1) Assert every point covered by Data1 is also covered by any range in Data2
    assert (forall p: int :: 
                (p >= Data1StartIdx && p < Data1StartIdx + Data1Size) ==>
                (exists j: int :: (j >= 0 && j < N2) && (p >= Data2StartIndices[j] && p < Data2StartIndices[j] + Data2Sizes[j])));

    // 2) Assert no overlap in the ranges of Data2
    assert (forall i: int, j: int :: 
                (i >= 0 && i < N2 && j >= 0 && j < N2 && i != j) ==>
                (Data2StartIndices[i] + Data2Sizes[i] <= Data2StartIndices[j] || Data2StartIndices[j] + Data2Sizes[j] <= Data2StartIndices[i]));
}

procedure CheckRangesCoverTheSameVariable()
{
    var Size: int;

    var Data1StartIdx: int;
    var Data1Size: int;
    var Data2StartIndices: [int]int;
    var Data2Sizes: [int]int;

    var N2: int;

    N2 := 2;
    Data1StartIdx := 0; Data1Size := Size;

    Data2StartIndices[0] := 0; Data2Sizes[0] := Size div 2;
    Data2StartIndices[1] := Size div 2; Data2Sizes[1] := Size div 2;

    // 1) Assert every point covered by Data1 is also covered by any range in Data2
    assert (forall p: int :: 
                (p >= Data1StartIdx && p < Data1StartIdx + Data1Size) ==>
                (exists j: int :: (j >= 0 && j < N2) && (p >= Data2StartIndices[j] && p < Data2StartIndices[j] + Data2Sizes[j])));

    // 2) Assert no overlap in the ranges of Data2
    assert (forall i: int, j: int :: 
                (i >= 0 && i < N2 && j >= 0 && j < N2 && i != j) ==>
                (Data2StartIndices[i] + Data2Sizes[i] <= Data2StartIndices[j] || Data2StartIndices[j] + Data2Sizes[j] <= Data2StartIndices[i]));
}

procedure CheckData1CoveredByData2AndNoOverlap(n: int)
    requires n > 0; // Ensures 'n' is positive, relevant if 'n' is used in division
{
    var Data1StartIdx, Data1Size, N2: int;
    var Data2StartIndices: [int]int;
    var Data2Sizes: [int]int;

    Data1StartIdx := 0;
    Data1Size := n;

    // Assuming Data2's size and start are expressions of 'n'
    N2 := 2; // Given Data2 can have multiple ranges. Adjust as necessary for more ranges

    // Initialize Data2 based on expressions involving 'n'
    Data2StartIndices[0] := 0;
    Data2Sizes[0] := n div 2; // Rounded down is default behavior

    Data2StartIndices[1] := n div 2;
    Data2Sizes[1] := n div 2; // Handles odd 'n' elegantly by covering the whole range

    // 1) Every point in Data1's range is also covered by Data2's ranges
    assert (forall p: int ::
                (p >= Data1StartIdx && p < Data1StartIdx + Data1Size) ==>
                (exists j: int :: (j >= 0 && j < N2) && (p >= Data2StartIndices[j] && p < Data2StartIndices[j] + Data2Sizes[j])));

    // 2) Check for no overlap in Data2's ranges
    // This is simple given the structured initialization but becomes more complex with more ranges
    assert (forall i: int, j: int ::
                (i >= 0 && i < N2 && j >= 0 && j < N2 && i != j) ==>
                (Data2StartIndices[i] + Data2Sizes[i] <= Data2StartIndices[j] || Data2StartIndices[j] + Data2Sizes[j] <= Data2StartIndices[i]));
}

procedure CheckExactCoverage()
{
    var N1, N2: int;
    var Data1StartIndices: [int]int;
    var Data1Sizes: [int]int;
    var Data2StartIndices: [int]int;
    var Data2Sizes: [int]int;

    N1 := 2; // Example: Total ranges in Data1
    N2 := 3; // Example: Total ranges in Data2

	Data1StartIndices[0] := 0;
	Data1Sizes[0] := 5;
	Data1StartIndices[1] := 5;
	Data1Sizes[1] := 5;

	Data2StartIndices[0] := 0;
	Data2Sizes[0] := 5;
	Data2StartIndices[1] := 5;
	Data2Sizes[1] := 4;
	Data2StartIndices[2] := 9;
	Data2Sizes[2] := 1;

    // Assert every point covered by Data1 is also covered by Data2 and vice versa
    assert (forall p: int :: 
                ((exists i: int :: (i >= 0 && i < N1) && (p >= Data1StartIndices[i] && p < Data1StartIndices[i] + Data1Sizes[i])) ==>
                 (exists j: int :: (j >= 0 && j < N2) && (p >= Data2StartIndices[j] && p < Data2StartIndices[j] + Data2Sizes[j]))));

    assert (forall p: int :: 
                ((exists j: int :: (j >= 0 && j < N2) && (p >= Data2StartIndices[j] && p < Data2StartIndices[j] + Data2Sizes[j])) ==>
                 (exists i: int :: (i >= 0 && i < N1) && (p >= Data1StartIndices[i] && p < Data1StartIndices[i] + Data1Sizes[i]))));
}
