func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) -> memref<?x?xf32> {
     %index_0 = arith.constant 0: index
     %index_1 = arith.constant 1: index
     %I = memref.dim %A, %index_0 : memref<?x?xf32>
     %J = memref.dim %B, %index_1 : memref<?x?xf32>
     %K = memref.dim %A, %index_1 : memref<?x?xf32>
     %T_i = arith.constant 4: index  // Tile size for i
     %T_j = arith.constant 8: index  // Tile size for j
     %T_k = arith.constant 16: index // Tile size for k
     
     // Compute number of tiles for i
     %num_tiles_i = arith.divui %I, %T_i : index
     %len_i = arith.muli %num_tiles_i, %T_i : index
     %has_partial_tile_i = arith.cmpi ne, %I, %len_i : index
     %extra_i = arith.select %has_partial_tile_i, %index_1, %index_0 : index
     %total_tiles_i = arith.addi %num_tiles_i, %extra_i : index
     
     // Compute number of tiles for j
     %num_tiles_j = arith.divui %J, %T_j : index
     %len_j = arith.muli %num_tiles_j, %T_j : index
     %has_partial_tile_j = arith.cmpi ne, %J, %len_j : index
     %extra_j = arith.select %has_partial_tile_j, %index_1, %index_0 : index
     %total_tiles_j = arith.addi %num_tiles_j, %extra_j : index
     
     // Compute number of tiles for k
     %num_tiles_k = arith.divui %K, %T_k : index
     %len_k = arith.muli %num_tiles_k, %T_k : index
     %has_partial_tile_k = arith.cmpi ne, %K, %len_k: index
     %extra_k = arith.select %has_partial_tile_k, %index_1, %index_0 : index
     %total_tiles_k = arith.addi %num_tiles_k, %extra_k : index
     
     scf.for %tile_i = %index_0 to %total_tiles_i step %index_1 {
       %start_i = arith.muli %tile_i, %T_i : index
       %end_i = arith.addi %start_i, %T_i : index
       // Bound check for the last tile in i
       %tile_end_i = arith.minui %end_i, %I : index
       
       scf.for %tile_j = %index_0 to %total_tiles_j step %index_1 {
         %start_j = arith.muli %tile_j, %T_j : index
         %end_j = arith.addi %start_j, %T_j : index
         // Bound check for the last tile in j
         %tile_end_j = arith.minui %end_j, %J : index
         
         scf.for %tile_k = %index_0 to %total_tiles_k step %index_1 {
           %start_k = arith.muli %tile_k, %T_k : index
           %end_k = arith.addi %start_k, %T_k : index
           // Bound check for the last tile in k
           %tile_end_k = arith.minui %end_k, %K : index

           scf.for %i = %start_i to %tile_end_i step %index_1 {
             scf.for %j = %start_j to %tile_end_j step %index_1 {
            	scf.for %k = %index_0 to %K step %index_1 {
              	    %a_val = memref.load %A[%i, %k] : memref<?x?xf32>
              	    %c_val = memref.load %C[%i, %k] : memref<?x?xf32>
              		%b_val = memref.load %B[%k, %j] : memref<?x?xf32>
              		%product = arith.mulf %a_val, %b_val : f32
              		%accum_next = arith.addf %c_val, %product : f32
              		memref.store %accum_next, %C[%i, %j] : memref<?x?xf32>
             }   
           }   
         }   
       }   
      }
     }   
     return %C : memref<?x?xf32>
 }
