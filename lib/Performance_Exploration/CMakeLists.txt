cmake_minimum_required(VERSION 3.10)
# Set your compiler
set(CMAKE_C_COMPILER clang-18)

# Paths
set(MLIR_PATH /home/lib/llvm-project/build)
set(MLIR_LIB_PATH ${MLIR_PATH}/lib)

# Compilation flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffp-contract=off -fopenmp -fopenmp-extensions -I/usr/lib/llvm-18/lib/clang/18/include -Wall -O3 -fPIC")

# Linker flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -L/usr/lib/llvm-18/lib -L${MLIR_LIB_PATH} -lmlir_runner_utils -lmlir_c_runner_utils -lblas -Wl,-rpath,${MLIR_LIB_PATH}")

# Define a function to add targets with custom commands
function(add_mlir_target target_name source_c source_mlir)
	set(LLVM_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.mlir.ll.o")
	set(LLVM_LL "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.ll")
	set(MLIR_LL "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.mlir.ll")
	set(MLIR_TMP_LL "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_tmp.llvm.mlir")
	set(MLIR_LLVM_MLIR "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.llvm.mlir")
	add_executable(${target_name} ${source_c} ${LLVM_OBJ})
	
	set(TIMEOUT 20)  # Set the timeout in seconds

	add_custom_command(
		OUTPUT ${LLVM_OBJ}
		COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/run_with_timeout.py ${TIMEOUT} ${CMAKE_C_COMPILER} -O3 -c ${MLIR_LL} -o ${LLVM_OBJ}
		DEPENDS ${MLIR_LL}
		COMMENT "Converting ${MLIR_LL} to ${LLVM_OBJ}"
	)

	add_custom_command(
		OUTPUT ${LLVM_LL}
		COMMAND ${CMAKE_C_COMPILER} -S -emit-llvm ${CMAKE_C_FLAGS} ${source_c} -o ${LLVM_LL}
		DEPENDS ${source_c}
		COMMENT "Generating LLVM IR from ${source_c}"
	)

	add_custom_command(
		OUTPUT ${MLIR_LL}
		COMMAND ${MLIR_PATH}/bin/mlir-translate --mlir-to-llvmir ${MLIR_LLVM_MLIR} > ${MLIR_LL}
		DEPENDS ${MLIR_LLVM_MLIR}
		COMMENT "Translating ${MLIR_LLVM_MLIR} to LLVM IR"
	)

	add_custom_command(
		OUTPUT ${MLIR_LLVM_MLIR}
		COMMAND ${MLIR_PATH}/bin/mlir-opt ${MLIR_TMP_LL} --test-transform-dialect-erase-schedule > ${MLIR_LLVM_MLIR}
		DEPENDS ${MLIR_TMP_LL}
		COMMENT "Running mlir-opt on ${MLIR_TMP_LL}"
	)


	add_custom_command(
		OUTPUT ${MLIR_TMP_LL}
		COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${source_mlir} ${CMAKE_BINARY_DIR}/${source_mlir}
		COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/run_with_timeout.py ${TIMEOUT} ${MLIR_PATH}/bin/mlir-transform-opt ${source_mlir} > ${MLIR_TMP_LL}
		COMMENT "Running mlir-transform-opt on ${source_mlir} with a timeout of ${TIMEOUT} seconds"
	)

	add_custom_target(${target_name}_prepare ALL DEPENDS ${MLIR_TMP_LL} ${MLIR_LLVM_MLIR} ${MLIR_LL} ${LLVM_OBJ})
	add_dependencies(${target_name} ${target_name}_prepare)
endfunction()

# Add targets
add_mlir_target(batch_matmul batch_matmul.c batch_matmul.mlir)
add_mlir_target(search_batch_matmul search_batch_matmul.c specialized_transform.mlir)

# Clean target
add_custom_target(clean-all
	COMMAND ${CMAKE_COMMAND} -E rm -f matmul matmul.llvm.mlir matmul.ll matmul_tmp.llvm.mlir matmul.mlir.ll
)