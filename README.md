# Ensuring Matrix Multiplication Correctness in MLIR using Boogie

We introduce a loop DSL (Domain Specific Language) for representing loop data like the start index, stride, and nesting patterns. Any MLIR kernel can be deterministically lowered into our loop DSL. The design of the loop DSL aims to capture the maximum information without introducing complexity that might require hand-crafted invariants, crucial for push-button verification and real usability for developers. The loop DSL then generates Boogie code with various assertions attempting to verify the correctness of the tiling.


## Running locally
```sh
$ export BOOGIE="/path/to/boogie/3.1.4"
$ pip install -r requirements.txt
$ python3 loopcc.py test/matmul.loop test/tile.loop
```
