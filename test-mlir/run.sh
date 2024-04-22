#!bash
set -x
#Lower to LLVM dialect
($MLIR_PATH/bin/mlir-opt $1.mlir -one-shot-bufferize="bufferize-function-boundaries=1 allow-return-allocs" -drop-equivalent-buffer-results | $MLIR_PATH/bin/mlir-opt -convert-linalg-to-loops -convert-scf-to-cf  -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm  -convert-func-to-llvm -reconcile-unrealized-casts | $MLIR_PATH/bin/mlir-translate -mlir-to-llvmir) > $1.ll
#Create optimized LLVM bytecode
$MLIR_PATH/bin/opt -O3 $1.ll -o $1.bc
#Create object code
$MLIR_PATH/bin/llc -filetype=obj $1.bc
#Create and run baseline plain-C version
gcc -Dbaseline=1 -O3 -o full $2.c; ./full
#Create and run version with MLIR-generated function
gcc -Dbaseline=0 -O3 -o full $2.c $1.o; ./full
