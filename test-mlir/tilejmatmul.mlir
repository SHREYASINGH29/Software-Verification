func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) -> memref<?x?xf32> {
    %index_0 = arith.constant 0: index
    %index_1 = arith.constant 1: index
    %I = memref.dim %A, %index_0 : memref<?x?xf32>
    %J = memref.dim %B, %index_1 : memref<?x?xf32>
    %K = memref.dim %A, %index_1 : memref<?x?xf32>
    %T = arith.constant 4: index  // Example tile size
    
    // Compute number of tiles for j loop
    %num_tiles_j = arith.divui %J, %T : index
    %len = arith.muli %num_tiles_j, %T : index
    %has_partial_tile_j = arith.cmpi ne, %J, %len : index
    %extra_j = arith.select %has_partial_tile_j, %index_1, %index_0 : index
    %total_tiles_j = arith.addi %num_tiles_j, %extra_j : index
    
    scf.for %i = %index_0 to %I step %index_1 {
      scf.for %tile_j = %index_0 to %total_tiles_j step %index_1 {
        %start_j = arith.muli %tile_j, %T : index
        %end_j = arith.addi %start_j, %T : index
        // Bound check for the last tile in j
        %tile_end_j = arith.minui %end_j, %J : index
      
        scf.for %j = %start_j to %tile_end_j step %index_1 {
          scf.for %k = %index_0 to %K step %index_1{
            %a_val = memref.load %A[%i, %k] : memref<?x?xf32>
            %b_val = memref.load %B[%k, %j] : memref<?x?xf32>
            %c_val = memref.load %C[%i, %j] : memref<?x?xf32>
            %product = arith.mulf %a_val, %b_val : f32 
            %sum1 = arith.addf %c_val, %product : f32 
            memref.store %sum1, %C[%i, %j] : memref<?x?xf32>
          }
        }
      }   
    }   
    return %C : memref<?x?xf32>
}
