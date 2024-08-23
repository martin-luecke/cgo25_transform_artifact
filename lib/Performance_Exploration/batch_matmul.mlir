module attributes {transform.with_named_sequence} {
  func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
  func.func private @libsxmm_smm(memref<68x68xf32, strided<[68, 68], offset: 68>>,memref<68x68xf32, strided<[68, 68], offset: 68>>,memref<68x68xf32, strided<[68, 68], offset: 68>>) attributes { llvm.emit_c_interface }

  func.func @matmul_mlir(%C: memref<6x196x256xf32>, %A: memref<6x196x2304xf32>, %B: memref<6x2304x256xf32>) attributes {llvm.emit_c_interface} {
    linalg.batch_matmul {library_call="libsxmm_smm"} ins(%A, %B : memref<6x196x2304xf32>, memref<6x2304x256xf32>) outs(%C : memref<6x196x256xf32>)
    return
  }

  transform.named_sequence @lower(%module: !transform.any_op {transform.consumed}) -> !transform.any_op {
    %module_1 = transform.apply_registered_pass "canonicalize" to %module : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %module_1 {
      transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerreduction"
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    transform.apply_patterns to %module_1 {
      transform.apply_patterns.vector.lower_transfer
    } : !transform.any_op
    transform.apply_patterns to %module_1 {
       transform.apply_patterns.vector.transfer_to_scf max_transfer_rank=1
    } : !transform.any_op
    transform.bufferization.buffer_loop_hoisting %module_1 : !transform.any_op
    transform.apply_patterns to %module_1 {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    %module_2 = transform.apply_registered_pass "convert-linalg-to-loops" to %module_1 : (!transform.any_op) -> !transform.any_op
    %module_3 = transform.apply_registered_pass "convert-scf-to-cf" to %module_2 : (!transform.any_op) -> !transform.any_op
    %module_4 = transform.apply_registered_pass "lower-affine" to %module_3 : (!transform.any_op) -> !transform.any_op
    %module_5 = transform.apply_registered_pass "convert-vector-to-llvm" to %module_4 : (!transform.any_op) -> !transform.any_op
    %module_6 = transform.apply_registered_pass "convert-math-to-llvm" to %module_5 : (!transform.any_op) -> !transform.any_op
    %module_7 = transform.apply_registered_pass "expand-strided-metadata" to %module_6 : (!transform.any_op) -> !transform.any_op
    %module_8 = transform.apply_registered_pass "lower-affine" to %module_7 : (!transform.any_op) -> !transform.any_op
    %module_9 = transform.apply_registered_pass "finalize-memref-to-llvm" to %module_8 : (!transform.any_op) -> !transform.any_op
    %module_10 = transform.apply_registered_pass "convert-func-to-llvm" to %module_9 : (!transform.any_op) -> !transform.any_op
    %module_11 = transform.apply_registered_pass "convert-index-to-llvm" to %module_10 : (!transform.any_op) -> !transform.any_op
    %module_12 = transform.apply_registered_pass "reconcile-unrealized-casts" to %module_11 : (!transform.any_op) -> !transform.any_op
    transform.yield %module_12 : !transform.any_op
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %module0 = transform.apply_registered_pass "linalg-generalize-named-ops" to %module : (!transform.any_op) -> !transform.any_op
    %batch_matmul = transform.structured.match ops{["linalg.generic"]} in %module0 : (!transform.any_op) -> !transform.any_op
    %batch_matmul_main, %split_off = transform.structured.split %batch_matmul after 192 { dimension = 1 } : !transform.any_op
    %tiled_op, %loops:3 = transform.structured.tile_using_for %batch_matmul_main [1,32,32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %inner, %vect_loops:4 = transform.structured.tile_using_for %tiled_op [1,1,1,16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %inner : !transform.any_op
    %module_1 = transform.apply_registered_pass "canonicalize" to %module0 : (!transform.any_op) -> !transform.any_op
    %lowered = transform.include @lower failures(propagate) (%module_1) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
