source_filename = "shfl_up_sync.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@shfl_up_sync_shared_var_arr = extern_weak dso_local addrspace(1) global [32 * i32]
@shfl_up_sync_updated = extern_weak dso_local addrspace(1) global [32 * i8]

; functions are __device__ by default
; shfl_up_sync(unsigned mask, unsigned var, unsigned int delta, int width=warpSize)
; llvm does not treat signed and unsigned ints differently
define i32 @shfl_up_sync(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) {
    ; copy over arguments
    %4 = alloca i32, align 4
    %5 = alloca i32, align 4
    %6 = alloca i32, align 4
    %7 = alloca i32, align 4
    store i32 %0, ptr %4, align 4
    store i32 %1, ptr %5, align 4
    store i32 %2, ptr %6, align 4
    store i32 %3, ptr %7, align 4

    ; tid.x
    %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    %9 = getelementptr inbounds 

    ; assign values to global arrays
    store i32 %1, 

}