#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// M , N , K
//196,256,2304
//196x2304 * 2304x256

module attributes {transform.with_named_sequence} {
  func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
  func.func @matmul_mlir(%A: memref<6x196x2304xf32>, %B: memref<6x2304x256xf32>) -> (memref<6x196x256xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %f0 = arith.constant 0.0 : f32
    %b = memref.dim %A, %c0 : memref<6x196x2304xf32>
    %x = memref.dim %A, %c1 : memref<6x196x2304xf32>
    %y = memref.dim %B, %c2 : memref<6x2304x256xf32>
    %C = memref.alloc() : memref<6x196x256xf32>
    linalg.batch_matmul {library_call="libsxmm_smm"} ins(%A, %B : memref<6x196x2304xf32>, memref<6x2304x256xf32>) outs(%C : memref<6x196x256xf32>)
    return %C : memref<6x196x256xf32>
  }

  transform.named_sequence @lower(%module: !transform.any_op {transform.consumed}) -> !transform.any_op {
    %module_1 = transform.apply_registered_pass "canonicalize" to %module : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %module_1 {
      transform.apply_patterns.vector.lower_multi_reduction
      transform.apply_patterns.vector.lower_contraction
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.apply_patterns to %module_1 {
      transform.apply_patterns.vector.transfer_to_scf
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

    // Tile the main computation
    %matmul = transform.structured.match ops{["linalg.generic"]} in %module0 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %loops:4 = transform.structured.tile_using_for %matmul [tile0,tile1,tile2,tile3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)


    %scope = transform.get_parent_op %tiled_op {op_name="func.func", deduplicate} : (!transform.any_op) -> !transform.any_op
    transform.alternatives %scope : !transform.any_op {
      ^bb0(%arg0: !transform.any_op):
      %tune_var = transform.param.constant do_vect : i64 -> !transform.param<i64>
      %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %tune_var, %c1 : !transform.param<i64>
      %generic = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %generic : !transform.any_op
    }, {
      ^bb0(%arg0: !transform.any_op):
    }
    %lowered = transform.include @lower failures(propagate) (%module0) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}