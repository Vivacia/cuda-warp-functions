#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

using namespace llvm;

define uint34_t @get_thread_id() {
  %tid = call spir_func i32 @__builtin_zext_i64_i32(i64 %ThreadIdInWorkGroup)
  ret %tid
}


int main() {
  LLVMContext context;
  IRBuilder<> builder(context);
  Module module("shuffle_up", context);

  // Define function signature
  Type *float_ty = Type::getFloatTy(context);
  FunctionType *shuffle_up_ty = FunctionType::get(
      Type::getVoidTy(context), {float_ty->getPointerTo(), float_ty->getPointerTo()}, false);
  Function *shuffle_up_fn =
      Function::Create(shuffle_up_ty, Function::ExternalLinkage, "shuffle_up", &module);

  // Create basic block
  BasicBlock *entry_bb = BasicBlock::Create(context, "entry", shuffle_up_fn);
  builder.SetInsertPoint(entry_bb);

  // Get the lane ID within the warp
  Function *read_reg_fn = Intrinsic::getDeclaration(&module, Intrinsic::read_register);
  Value *tid =  @get_thread_id();
  Value *laneid = builder.CreateAnd(tid, 31);

  // Allocate shared memory for input and output
  Value *shared = builder.CreateAlloca(ArrayType::get(float_ty, 32), ConstantInt::get(Type::getInt32Ty(context), 16), "shared");

  // Load input into shared memory
  Value *input = shuffle_up_fn->arg_begin();
  Value *ptr = builder.CreateGEP(shared, {ConstantInt::get(Type::getInt32Ty(context), 0), laneid});
  Value *val = builder.CreateLoad(float_ty, input);
  builder.CreateStore(val, ptr);
  barrier(CLK_LOCAL_MEM_FENCE);

  // Shuffle up operation
  Value *dstidx = builder.CreateAdd(laneid, ConstantInt::get(Type::getInt32Ty(context), -1));
  Value *dstptr = builder.CreateGEP(shared, {ConstantInt::get(Type::getInt32Ty(context), 0), dstidx});
  Value *srcptr = builder.CreateGEP(shared, {ConstantInt::get(Type::getInt32Ty(context), 0), laneid});
  Value *dstval = builder.CreateLoad(float_ty, dstptr);
  Value *srcval = builder.CreateLoad(float_ty, srcptr);
  Value *res = builder.CreateSelect(builder.CreateICmpEQ(laneid, ConstantInt::get(Type::getInt32Ty(context), 0)), ConstantFP::get(float_ty, 0.0), dstval);
  builder.CreateStore(res, dstptr);
  builder.CreateStore(srcval, shuffle_up_fn->arg_begin() + 1);
  barrier(CLK_LOCAL_MEM_FENCE)

  builder.CreateRetVoid();

  // Verify module and print IR
  verifyModule(module, &errs());
  module.print(outs(), nullptr);

  return 0;
}
