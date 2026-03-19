#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include <cstdlib>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MEMORYOPTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

// TODO: use enums to access struct fields.
struct ExtractMemRefOpConversion : public OpConversionPattern<ExtractMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value tensorPtrStruct = rewriter.getRemappedValue(op.getSrc());
    auto memRefTy = cast<MemRefType>(op.getType());
    auto rank = memRefTy.getRank();
    auto memRefStructTy = getTypeConverter()->convertType(op.getType());
    auto memRefStructFields =
        cast<LLVM::LLVMStructType>(memRefStructTy).getBody();
    auto i64Ty = IntegerType::get(getContext(), 64);

    auto copyValue = [&](Value to, int64_t idxFrom, int64_t idxTo) {
      auto valueTy = memRefStructFields[idxTo];
      Value val = LLVM::ExtractValueOp::create(rewriter, loc, valueTy,
                                               tensorPtrStruct, idxFrom);
      return LLVM::InsertValueOp::create(rewriter, loc, memRefStructTy, to, val,
                                         idxTo);
    };

    Value res = b.undef(memRefStructTy);
    // Copy base.
    res = copyValue(res, 0, 1);
    // Use 0 offset.
    res = LLVM::InsertValueOp::create(rewriter, loc, memRefStructTy, res,
                                      b.i64_val(0), 2);
    // Copy shape.
    res = copyValue(res, 2, 3);
    // Copy strides.
    res = copyValue(res, 3, 4);

    rewriter.replaceOp(op, res);

    return success();
  }
};

struct ExtractIndicesOpConversion
    : public OpConversionPattern<ExtractIndicesOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractIndicesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value tensorPtrStruct = rewriter.getRemappedValue(op.getSrc());
    auto rank = op.getNumResults();
    auto i64Ty = IntegerType::get(getContext(), 64);
    SmallVector<Value> indices;

    for (int64_t i = 0; i < rank; i++) {
      indices.push_back(
          LLVM::ExtractValueOp::create(rewriter, loc, i64Ty, tensorPtrStruct,
                                       SmallVector<int64_t, 2>{1, i}));
    }

    rewriter.replaceOp(op, indices);

    return success();
  }
};

struct PtrToMemRefOpConversion : public OpConversionPattern<PtrToMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PtrToMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value ptr = rewriter.getRemappedValue(op.getSrc());
    auto memRefStructTy = getTypeConverter()->convertType(op.getType());

    Value res = b.undef(memRefStructTy);
    res =
        LLVM::InsertValueOp::create(rewriter, loc, memRefStructTy, res, ptr, 1);
    rewriter.replaceOp(op, res);

    return success();
  }
};

struct MakeTensorPtrOpConversion : public OpConversionPattern<MakeTensorPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto structTy = getTypeConverter()->convertType(op.getType());
    auto i64Ty = IntegerType::get(getContext(), 64);

    auto insertArray = [&](Value structVal, auto values, int64_t idx,
                           Type zextTo = nullptr) {
      for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        Value val = values[i];
        if (zextTo)
          val = LLVM::ZExtOp::create(rewriter, loc, zextTo, val);
        structVal =
            LLVM::InsertValueOp::create(rewriter, loc, structTy, structVal, val,
                                        SmallVector<int64_t, 2>{idx, i});
      }
      return structVal;
    };

    Value res = b.undef(structTy);
    // 0 - base pointer.
    auto base = rewriter.getRemappedValue(op.getBase());
    res = LLVM::InsertValueOp::create(rewriter, loc, structTy, res, base,
                                      ArrayRef<int64_t>{0});
    // 1 - array<rank> for offsets. Promote values to i64.
    res = insertArray(res, op.getOffsets(), 1, i64Ty);
    // 2 - array<rank> for shape.
    res = insertArray(res, op.getShape(), 2);
    // 3 - array<rank> for strides.
    res = insertArray(res, op.getStrides(), 3);

    rewriter.replaceOp(op, res);

    return success();
  }
};

struct AdvanceOpConversion : public OpConversionPattern<AdvanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i64Ty = IntegerType::get(getContext(), 64);
    Value res = rewriter.getRemappedValue(op.getPtr());
    Type structTy = res.getType();
    auto offsets = op.getOffsets();

    for (int64_t i = 0; i < offsets.size(); ++i) {
      auto oldOffset = LLVM::ExtractValueOp::create(
          rewriter, loc, i64Ty, res, SmallVector<int64_t, 2>{1, i});
      auto step = LLVM::SExtOp::create(rewriter, loc, i64Ty, offsets[i]);
      auto newOffset = LLVM::AddOp::create(rewriter, loc, oldOffset, step);
      res = LLVM::InsertValueOp::create(rewriter, loc, structTy, res, newOffset,
                                        SmallVector<int64_t, 2>{1, i});
    }

    rewriter.replaceOp(op, res);

    return success();
  }
};

struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type ptrTy = LLVM::LLVMPointerType::get(getContext());
    Value ptr = rewriter.getRemappedValue(op.getPtr());
    Type resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resTy, ptr, 0,
                                              op.getIsVolatile());
    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value ptr = rewriter.getRemappedValue(op.getPtr());
    Value val = rewriter.getRemappedValue(op.getValue());
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, val, ptr);
    return success();
  }
};


struct VectorMaskedLoadOpConversion : public OpConversionPattern<vector::MaskedLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *typeConverter = getTypeConverter();
    Type ptrTy = LLVM::LLVMPointerType::get(getContext());
    Type resTy = getTypeConverter()->convertType(op.getType());
    Value memref = adaptor.getBase();
    Value ptr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, memref, ArrayRef<int64_t>{1});

    auto vecTy = mlir::dyn_cast<VectorType>(resTy);
    if (!vecTy) {
      return failure();
    }
    auto i64Ty = IntegerType::get(getContext(), 64);
    auto i32Ty = IntegerType::get(getContext(), 32);
    auto i1Ty = IntegerType::get(getContext(), 1);
    auto one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, resTy, one, /*alignment=*/16);
    auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));

    Block *currentBlock = rewriter.getBlock();
    Block *continueBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *headerBlock = rewriter.createBlock(currentBlock->getParent(), continueBlock->getIterator());
    headerBlock->addArgument(i64Ty, loc);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{zero}, headerBlock);
    rewriter.setInsertionPointToStart(headerBlock);

    int64_t vecSize = vecTy.getNumElements();
    auto constVecSize = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(vecSize));
    auto constVecSizeExt = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(vecSize));
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    Value nElements = funcOp.getArgument(3);
    Value pidX = funcOp.getArgument(4);
    Value blockStart = LLVM::MulOp::create(rewriter, loc, i32Ty, pidX, constVecSize);
    auto blockStartExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, blockStart);
    auto nElementsExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, nElements);
    Value remaining = LLVM::SubOp::create(rewriter, loc, i64Ty, nElementsExt, blockStartExt);
    auto cmpVecSize = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::slt, remaining, constVecSizeExt);
    Value clampElements = LLVM::SelectOp::create(rewriter, loc, i64Ty, cmpVecSize, remaining, constVecSizeExt);
    auto two = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(2));
    auto three = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(3));
    Value iv = headerBlock->getArgument(0);
    auto remain = LLVM::SubOp::create(rewriter, loc, i64Ty, clampElements, iv);
    auto vl = LLVM::CallIntrinsicOp::create(rewriter, loc, i64Ty,
                rewriter.getStringAttr("llvm.riscv.vsetvli.i64"),
                ArrayRef<Value>{remain, two, three}).getResult(0);
    auto vlTrunc = LLVM::TruncOp::create(rewriter, loc, i32Ty, vl);

    Type elemTy = vecTy.getElementType();
    auto loadPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, ptr,
                    ArrayRef<LLVM::GEPArg>{iv});//,
                    //LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nuw);
    VectorType nxv16f32Ty = VectorType::get({16}, rewriter.getF32Type(), /*scalable=*/true);
    VectorType nxv16i1Ty = VectorType::get({16}, i1Ty, /*scalable=*/true);
    auto splatAttr = SplatElementsAttr::get(nxv16i1Ty, rewriter.getBoolAttr(true));
    auto allTrueMask = LLVM::ConstantOp::create(rewriter, loc, nxv16i1Ty, splatAttr);
    auto emptyDict = rewriter.getDictionaryAttr({});
    auto alignAttr = rewriter.getNamedAttr("llvm.align", rewriter.getI64IntegerAttr(4));
    auto ptrAttrDict = DictionaryAttr::get(rewriter.getContext(), {alignAttr});
    auto vpLoadOp = LLVM::CallIntrinsicOp::create(rewriter, loc, nxv16f32Ty,
                    rewriter.getStringAttr("llvm.vp.load.nxv16f32.p0"),
                    ArrayRef<Value>{loadPtr, allTrueMask, vlTrunc});
    //vpLoadOp.setArgAttrsAttr(rewriter.getArrayAttr({ptrAttrDict, emptyDict, emptyDict}));
    Value vpLoad = vpLoadOp.getResult(0);

    Value destPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, alloca,
                    ArrayRef<LLVM::GEPArg>{iv});//,
                    //LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nuw);
    auto vpStoreOp = LLVM::CallIntrinsicOp::create(rewriter, loc,
                    rewriter.getStringAttr("llvm.vp.store.nxv16f32.p0"),
                    ValueRange({vpLoad, destPtr, allTrueMask, vlTrunc}));
    /*auto dict = adaptor.getBase().getDefiningOp()->getAttrs();
    SmallVector<NamedAttribute, 1> refineAttr{alignAttr};
    for (auto attr : dict) {
        if (attr.getName().str() != "llvm.nonnull")
            refineAttr.push_back(attr);
    }
    auto refineDict = DictionaryAttr::get(rewriter.getContext(), refineAttr);
    vpStoreOp.setArgAttrsAttr(rewriter.getArrayAttr({emptyDict, refineDict, emptyDict, emptyDict}));*/

    //auto vlZext = LLVM::ZExtOp::create(rewriter, loc, i64Ty, vl);
    auto iNext = LLVM::AddOp::create(rewriter, loc, i64Ty, iv, vl);
    auto done = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge, iNext, clampElements);
    LLVM::CondBrOp::create(rewriter, loc, done,
                                    continueBlock, ValueRange{},
                                    headerBlock, ValueRange{iNext});

    rewriter.setInsertionPointToStart(continueBlock);
    auto finalLoad = LLVM::LoadOp::create(rewriter, loc, resTy, alloca, /*alignment=*/16);

    rewriter.replaceOp(op, finalLoad);
    return success();
  }
};

struct VectorMaskedStoreOpConversion : public OpConversionPattern<vector::MaskedStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *typeConverter = getTypeConverter();
    Type ptrTy = LLVM::LLVMPointerType::get(getContext());
    Type valTy = typeConverter->convertType(op.getValueToStore().getType());
    Value memref = adaptor.getBase();
    Value valueToStore = adaptor.getValueToStore();

    Value ptr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, memref, ArrayRef<int64_t>{1});

    auto vecTy = mlir::dyn_cast<VectorType>(valTy);
    if (!vecTy) {
      return failure();
    }
    auto i64Ty = IntegerType::get(getContext(), 64);
    auto i32Ty = IntegerType::get(getContext(), 32);
    auto i1Ty = IntegerType::get(getContext(), 1);
    auto one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, valTy, one, /*alignment=*/16);
    auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));
    LLVM::StoreOp::create(rewriter, loc, valueToStore, alloca, /*alignment=*/16);

    Block *currentBlock = rewriter.getBlock();
    Block *continueBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *headerBlock = rewriter.createBlock(currentBlock->getParent(), continueBlock->getIterator());
    headerBlock->addArgument(i64Ty, loc);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{zero}, headerBlock);
    rewriter.setInsertionPointToStart(headerBlock);

    int64_t vecSize = vecTy.getNumElements();
    auto constVecSize = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(vecSize));
    auto constVecSizeExt = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(vecSize));
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    Value nElements = funcOp.getArgument(3);
    Value pidX = funcOp.getArgument(4);
    Value blockStart = LLVM::MulOp::create(rewriter, loc, i32Ty, pidX, constVecSize);
    auto blockStartExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, blockStart);
    auto nElementsExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, nElements);
    Value remaining = LLVM::SubOp::create(rewriter, loc, i64Ty, nElementsExt, blockStartExt);
    auto cmpVecSize = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::slt, remaining, constVecSizeExt);
    Value clampElements = LLVM::SelectOp::create(rewriter, loc, i64Ty, cmpVecSize, remaining, constVecSizeExt);
    auto two = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(2));
    auto three = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(3));
    Value iv = headerBlock->getArgument(0);
    auto remain = LLVM::SubOp::create(rewriter, loc, i64Ty, clampElements, iv);
    auto vl = LLVM::CallIntrinsicOp::create(rewriter, loc, i64Ty,
                rewriter.getStringAttr("llvm.riscv.vsetvli.i64"),
                ValueRange({remain, two, three})).getResult(0);
    auto vlTrunc = LLVM::TruncOp::create(rewriter, loc, i32Ty, vl);

    Type elemTy = vecTy.getElementType();
    auto loadPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, alloca,
                    ArrayRef<LLVM::GEPArg>{iv});//,
                    //LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nuw);
    VectorType nxv16f32Ty = VectorType::get({16}, rewriter.getF32Type(), /*scalable=*/true);
    VectorType nxv16i1Ty = VectorType::get({16}, i1Ty, /*scalable=*/true);
    auto splatAttr = SplatElementsAttr::get(nxv16i1Ty, rewriter.getBoolAttr(true));
    auto allTrueMask = LLVM::ConstantOp::create(rewriter, loc, nxv16i1Ty, splatAttr);
    //auto allTrueMask = rewriter.create<LLVM::ConstantOp>(loc, nxv16i1Ty, rewriter.getZeroAttr(nxv16i1Ty));
    auto emptyDict = rewriter.getDictionaryAttr({});
    auto alignAttr = rewriter.getNamedAttr("llvm.align", rewriter.getI64IntegerAttr(4));
    auto ptrAttrDict = DictionaryAttr::get(rewriter.getContext(), {alignAttr});
    auto vpLoadOp = LLVM::CallIntrinsicOp::create(rewriter, loc, nxv16f32Ty,
                    rewriter.getStringAttr("llvm.vp.load.nxv16f32.p0"),
                    ValueRange({loadPtr, allTrueMask, vlTrunc}));
    /*auto dict = adaptor.getBase().getDefiningOp()->getAttrs();
    SmallVector<NamedAttribute, 1> refineAttr{alignAttr};
    for (auto attr : dict) {
        if (attr.getName().str() != "llvm.nonnull")
            refineAttr.push_back(attr);
    }
    auto refineDict = DictionaryAttr::get(rewriter.getContext(), refineAttr);
    vpLoadOp.setArgAttrsAttr(rewriter.getArrayAttr({refineDict, emptyDict, emptyDict}));*/
    Value vpLoad = vpLoadOp.getResult(0);

    auto destPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, ptr,
                    ArrayRef<LLVM::GEPArg>{iv});//,
                    //LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nuw);
    auto vpStoreOp = LLVM::CallIntrinsicOp::create(rewriter, loc,
                    rewriter.getStringAttr("llvm.vp.store.nxv16f32.p0"),
                    ValueRange({vpLoad, destPtr, allTrueMask, vlTrunc}));
    //vpStoreOp.setArgAttrsAttr(rewriter.getArrayAttr({emptyDict, ptrAttrDict, emptyDict, emptyDict}));

    //auto vlZext = LLVM::ZExtOp::create(rewriter, loc, i64Ty, vl);
    auto iNext = LLVM::AddOp::create(rewriter, loc, i64Ty, iv, vl);
    auto done = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge, iNext, clampElements);
    LLVM::CondBrOp::create(rewriter, loc, done,
                                    continueBlock, ValueRange{},
                                    headerBlock, ValueRange{iNext});

    rewriter.setInsertionPointToStart(continueBlock);
    rewriter.eraseOp(op);

    return success();
  }
};

struct BranchMaskedLoadOpConversion : public OpConversionPattern<vector::MaskedLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *typeConverter = getTypeConverter();
    Type ptrTy = LLVM::LLVMPointerType::get(getContext());
    Type resTy = getTypeConverter()->convertType(op.getType());
    Value memref = adaptor.getBase();
    Value ptr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, memref, ArrayRef<int64_t>{1});

    auto vecTy = mlir::dyn_cast<VectorType>(resTy);
    if (!vecTy) {
      return failure();
    }
    auto i64Ty = IntegerType::get(getContext(), 64);
    auto i32Ty = IntegerType::get(getContext(), 32);
    auto i1Ty = IntegerType::get(getContext(), 1);
    //auto one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    //Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, resTy, one, /*alignment=*/16);
    //auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));

    /*Block *currentBlock = rewriter.getBlock();
    Block *continueBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *headerBlock = rewriter.createBlock(currentBlock->getParent(), continueBlock->getIterator());
    headerBlock->addArgument(i64Ty, loc);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{zero}, headerBlock);
    rewriter.setInsertionPointToStart(headerBlock);*/

    int64_t vecSize = vecTy.getNumElements();
    auto constVecSize = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(vecSize));
    auto constVecSizeExt = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(vecSize));
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    Value nElements = funcOp.getArgument(3);
    Value pidX = funcOp.getArgument(4);
    Value blockStart = LLVM::MulOp::create(rewriter, loc, i32Ty, pidX, constVecSize);
    auto blockStartExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, blockStart);
    auto nElementsExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, nElements);
    Value remaining = LLVM::SubOp::create(rewriter, loc, i64Ty, nElementsExt, blockStartExt);
    auto cmpVecSize = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ugt, remaining, constVecSizeExt);
    //Value clampElements = LLVM::SelectOp::create(rewriter, loc, i64Ty, cmpVecSize, remaining, constVecSizeExt);

    Block *currentBlock = rewriter.getBlock();
    Block *continueBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    continueBlock->addArgument(resTy, loc);
    Block *fullBlock = rewriter.createBlock(continueBlock->getParent(), continueBlock->getIterator());
    Block *partialBlock = rewriter.createBlock(continueBlock->getParent(), continueBlock->getIterator());
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, cmpVecSize, fullBlock, partialBlock);

    rewriter.setInsertionPointToStart(fullBlock);
    auto fullLoad = LLVM::LoadOp::create(rewriter, loc, resTy, ptr, /*alignment=*/16);
    LLVM::BrOp::create(rewriter, loc, ValueRange{fullLoad}, continueBlock);

    rewriter.setInsertionPointToStart(partialBlock);
    auto one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, resTy, one, /*alignment=*/16);
    auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));
    Block *headerBlock = rewriter.createBlock(partialBlock->getParent(), continueBlock->getIterator());
    headerBlock->addArgument(i64Ty, loc);
    Block *partialEndBlock = rewriter.createBlock(continueBlock->getParent(), continueBlock->getIterator());
    rewriter.setInsertionPointToEnd(partialBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{zero}, headerBlock);
    rewriter.setInsertionPointToStart(headerBlock);

    auto two = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(2));
    auto three = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(3));
    Value iv = headerBlock->getArgument(0);
    auto remain = LLVM::SubOp::create(rewriter, loc, i64Ty, remaining, iv);
    auto vl = LLVM::CallIntrinsicOp::create(rewriter, loc, i64Ty,
                rewriter.getStringAttr("llvm.riscv.vsetvli.i64"),
                ArrayRef<Value>{remain, two, three}).getResult(0);
    auto vlTrunc = LLVM::TruncOp::create(rewriter, loc, i32Ty, vl);

    Type elemTy = vecTy.getElementType();
    auto loadPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, ptr,
                    ArrayRef<LLVM::GEPArg>{iv});//,
                    //LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nuw);
    VectorType nxv16f32Ty = VectorType::get({16}, rewriter.getF32Type(), /*scalable=*/true);
    VectorType nxv16i1Ty = VectorType::get({16}, i1Ty, /*scalable=*/true);
    auto splatAttr = SplatElementsAttr::get(nxv16i1Ty, rewriter.getBoolAttr(true));
    auto allTrueMask = LLVM::ConstantOp::create(rewriter, loc, nxv16i1Ty, splatAttr);
    auto emptyDict = rewriter.getDictionaryAttr({});
    auto alignAttr = rewriter.getNamedAttr("llvm.align", rewriter.getI64IntegerAttr(4));
    auto ptrAttrDict = DictionaryAttr::get(rewriter.getContext(), {alignAttr});
    auto vpLoadOp = LLVM::CallIntrinsicOp::create(rewriter, loc, nxv16f32Ty,
                    rewriter.getStringAttr("llvm.vp.load.nxv16f32.p0"),
                    ArrayRef<Value>{loadPtr, allTrueMask, vlTrunc});
    //vpLoadOp.setArgAttrsAttr(rewriter.getArrayAttr({ptrAttrDict, emptyDict, emptyDict}));
    Value vpLoad = vpLoadOp.getResult(0);

    Value destPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, alloca,
                    ArrayRef<LLVM::GEPArg>{iv});//,
                    //LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nuw);
    auto vpStoreOp = LLVM::CallIntrinsicOp::create(rewriter, loc,
                    rewriter.getStringAttr("llvm.vp.store.nxv16f32.p0"),
                    ValueRange({vpLoad, destPtr, allTrueMask, vlTrunc}));
    /*auto dict = adaptor.getBase().getDefiningOp()->getAttrs();
    SmallVector<NamedAttribute, 1> refineAttr{alignAttr};
    for (auto attr : dict) {
        if (attr.getName().str() != "llvm.nonnull")
            refineAttr.push_back(attr);
    }
    auto refineDict = DictionaryAttr::get(rewriter.getContext(), refineAttr);
    vpStoreOp.setArgAttrsAttr(rewriter.getArrayAttr({emptyDict, refineDict, emptyDict, emptyDict}));*/

    //auto vlZext = LLVM::ZExtOp::create(rewriter, loc, i64Ty, vl);
    auto iNext = LLVM::AddOp::create(rewriter, loc, i64Ty, iv, vl);
    auto done = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge, iNext, remaining);
    LLVM::CondBrOp::create(rewriter, loc, done,
                                    partialEndBlock, ValueRange{},
                                    headerBlock, ValueRange{iNext});

    rewriter.setInsertionPointToStart(partialEndBlock);
    auto finalPartialLoad = LLVM::LoadOp::create(rewriter, loc, resTy, alloca, /*alignment=*/16);

    LLVM::BrOp::create(rewriter, loc, ValueRange{finalPartialLoad}, continueBlock);
    rewriter.setInsertionPointToStart(continueBlock);
    Value finalResult = continueBlock->getArgument(0);

    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

struct BranchMaskedStoreOpConversion : public OpConversionPattern<vector::MaskedStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *typeConverter = getTypeConverter();
    Type ptrTy = LLVM::LLVMPointerType::get(getContext());
    Type valTy = typeConverter->convertType(op.getValueToStore().getType());
    Value memref = adaptor.getBase();
    Value valueToStore = adaptor.getValueToStore();

    Value ptr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, memref, ArrayRef<int64_t>{1});

    auto vecTy = mlir::dyn_cast<VectorType>(valTy);
    if (!vecTy) {
      return failure();
    }
    auto i64Ty = IntegerType::get(getContext(), 64);
    auto i32Ty = IntegerType::get(getContext(), 32);
    auto i1Ty = IntegerType::get(getContext(), 1);
    //auto one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    //Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, valTy, one, /*alignment=*/16);
    //auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));
    //LLVM::StoreOp::create(rewriter, loc, valueToStore, alloca, /*alignment=*/16);

    /*Block *currentBlock = rewriter.getBlock();
    Block *continueBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *headerBlock = rewriter.createBlock(currentBlock->getParent(), continueBlock->getIterator());
    headerBlock->addArgument(i64Ty, loc);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{zero}, headerBlock);
    rewriter.setInsertionPointToStart(headerBlock);*/

    int64_t vecSize = vecTy.getNumElements();
    auto constVecSize = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(vecSize));
    auto constVecSizeExt = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(vecSize));
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    Value nElements = funcOp.getArgument(3);
    Value pidX = funcOp.getArgument(4);
    Value blockStart = LLVM::MulOp::create(rewriter, loc, i32Ty, pidX, constVecSize);
    auto blockStartExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, blockStart);
    auto nElementsExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, nElements);
    Value remaining = LLVM::SubOp::create(rewriter, loc, i64Ty, nElementsExt, blockStartExt);
    auto cmpVecSize = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ugt, remaining, constVecSizeExt);
    //Value clampElements = LLVM::SelectOp::create(rewriter, loc, i64Ty, cmpVecSize, remaining, constVecSizeExt);

    Block *currentBlock = rewriter.getBlock();
    Block *continueBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *fullBlock = rewriter.createBlock(continueBlock->getParent(), continueBlock->getIterator());
    Block *partialBlock = rewriter.createBlock(continueBlock->getParent(), continueBlock->getIterator());
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, cmpVecSize, fullBlock, partialBlock);

    rewriter.setInsertionPointToStart(fullBlock);
    LLVM::StoreOp::create(rewriter, loc, valueToStore, ptr, /*alignment=*/16);
    LLVM::BrOp::create(rewriter, loc, ValueRange{}, continueBlock);

    rewriter.setInsertionPointToStart(partialBlock);
    auto one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, valTy, one, /*alignment=*/16);
    LLVM::StoreOp::create(rewriter, loc, valueToStore, alloca, /*alignment=*/16);
    auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));
    Block *headerBlock = rewriter.createBlock(continueBlock->getParent(), continueBlock->getIterator());
    headerBlock->addArgument(i64Ty, loc);
    rewriter.setInsertionPointToEnd(partialBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{zero}, headerBlock);
    rewriter.setInsertionPointToStart(headerBlock);

    auto two = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(2));
    auto three = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(3));
    Value iv = headerBlock->getArgument(0);
    auto remain = LLVM::SubOp::create(rewriter, loc, i64Ty, remaining, iv);
    auto vl = LLVM::CallIntrinsicOp::create(rewriter, loc, i64Ty,
                rewriter.getStringAttr("llvm.riscv.vsetvli.i64"),
                ValueRange({remain, two, three})).getResult(0);
    auto vlTrunc = LLVM::TruncOp::create(rewriter, loc, i32Ty, vl);

    Type elemTy = vecTy.getElementType();
    auto loadPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, alloca,
                    ArrayRef<LLVM::GEPArg>{iv});//,
                    //LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nuw);
    VectorType nxv16f32Ty = VectorType::get({16}, rewriter.getF32Type(), /*scalable=*/true);
    VectorType nxv16i1Ty = VectorType::get({16}, i1Ty, /*scalable=*/true);
    auto splatAttr = SplatElementsAttr::get(nxv16i1Ty, rewriter.getBoolAttr(true));
    auto allTrueMask = LLVM::ConstantOp::create(rewriter, loc, nxv16i1Ty, splatAttr);
    //auto allTrueMask = rewriter.create<LLVM::ConstantOp>(loc, nxv16i1Ty, rewriter.getZeroAttr(nxv16i1Ty));
    auto emptyDict = rewriter.getDictionaryAttr({});
    auto alignAttr = rewriter.getNamedAttr("llvm.align", rewriter.getI64IntegerAttr(4));
    auto ptrAttrDict = DictionaryAttr::get(rewriter.getContext(), {alignAttr});
    auto vpLoadOp = LLVM::CallIntrinsicOp::create(rewriter, loc, nxv16f32Ty,
                    rewriter.getStringAttr("llvm.vp.load.nxv16f32.p0"),
                    ValueRange({loadPtr, allTrueMask, vlTrunc}));
    /*auto dict = adaptor.getBase().getDefiningOp()->getAttrs();
    SmallVector<NamedAttribute, 1> refineAttr{alignAttr};
    for (auto attr : dict) {
        if (attr.getName().str() != "llvm.nonnull")
            refineAttr.push_back(attr);
    }
    auto refineDict = DictionaryAttr::get(rewriter.getContext(), refineAttr);
    vpLoadOp.setArgAttrsAttr(rewriter.getArrayAttr({refineDict, emptyDict, emptyDict}));*/
    Value vpLoad = vpLoadOp.getResult(0);

    auto destPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, ptr,
                    ArrayRef<LLVM::GEPArg>{iv});//,
                    //LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nuw);
    auto vpStoreOp = LLVM::CallIntrinsicOp::create(rewriter, loc,
                    rewriter.getStringAttr("llvm.vp.store.nxv16f32.p0"),
                    ValueRange({vpLoad, destPtr, allTrueMask, vlTrunc}));
    //vpStoreOp.setArgAttrsAttr(rewriter.getArrayAttr({emptyDict, ptrAttrDict, emptyDict, emptyDict}));

    //auto vlZext = LLVM::ZExtOp::create(rewriter, loc, i64Ty, vl);
    auto iNext = LLVM::AddOp::create(rewriter, loc, i64Ty, iv, vl);
    auto done = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge, iNext, remaining);
    LLVM::CondBrOp::create(rewriter, loc, done,
                                    continueBlock, ValueRange{},
                                    headerBlock, ValueRange{iNext});

    rewriter.setInsertionPointToStart(continueBlock);
    rewriter.eraseOp(op);

    return success();
  }
};


struct PtrToIntOpConversion : public OpConversionPattern<triton::PtrToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = rewriter.getRemappedValue(op.getSrc());
    Type resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resTy, src);
    return success();
  }
};

struct IntToPtrOpConversion : public OpConversionPattern<triton::IntToPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = rewriter.getRemappedValue(op.getSrc());
    Type resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, resTy, src);
    return success();
  }
};

struct AddPtrOpConversion : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expect only scalar pointers here.
    assert(isa<PointerType>(op.getType()));
    auto ptrTy = cast<PointerType>(op.getPtr().getType());
    Type elemTy = getTypeConverter()->convertType(ptrTy.getPointeeType());
    Type resTy = getTypeConverter()->convertType(ptrTy);
    Value ptr = rewriter.getRemappedValue(op.getPtr());
    Value offset = rewriter.getRemappedValue(op.getOffset());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, resTy, elemTy, ptr, offset);
    return success();
  }
};

struct PtrBitcastConversion : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // By this moment we expect tt.bitcast used only for scalar pointer casts.
    // This cast becomes NOP for LLVM dialect, so simply return the source arg.
    assert(isa<PointerType>(op.getType()));
    assert(isa<PointerType>(op.getSrc().getType()));
    Value src = rewriter.getRemappedValue(op.getSrc());
    rewriter.replaceOp(op, src);
    return success();
  }
};

struct PtrSelectConversion : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // By this moment we expect tt.bitcast used only for scalar pointer casts.
    // This cast becomes NOP for LLVM dialect, so simply return the source arg.
    if (!isa<PointerType>(op.getType()))
      return failure();

    Value trueVal = rewriter.getRemappedValue(op.getTrueValue());
    Value falseVal = rewriter.getRemappedValue(op.getFalseValue());
    Value cond = rewriter.getRemappedValue(op.getCondition());
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, cond, trueVal, falseVal);
    return success();
  }
};

struct MemoryOpToLLVM
    : public triton::impl::MemoryOpToLLVMBase<MemoryOpToLLVM> {
  using MemoryOpToLLVMBase::MemoryOpToLLVMBase;

  MemoryOpToLLVM() : MemoryOpToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<ExtractMemRefOpConversion>(typeConverter, context);
    patterns.add<MakeTensorPtrOpConversion>(typeConverter, context);
    patterns.add<AdvanceOpConversion>(typeConverter, context);
    patterns.add<ExtractIndicesOpConversion>(typeConverter, context);
    patterns.add<LoadOpConversion>(typeConverter, context);
    patterns.add<StoreOpConversion>(typeConverter, context);

    char* vsetvl_env = getenv("TRITON_VSETVL_MINE");
    char* branch_env = getenv("TRITON_BRANCH_TAIL");
    if (branch_env) {
      llvm::outs()<<"!TRITON_BRANCH_TAIL is set\n";
      patterns.add<BranchMaskedLoadOpConversion>(typeConverter, context);
      patterns.add<BranchMaskedStoreOpConversion>(typeConverter, context);
    }
    else if (vsetvl_env) {
      llvm::outs()<<"!TRITON_VSETVL_MINE is set\n";
      patterns.add<VectorMaskedLoadOpConversion>(typeConverter, context);
      patterns.add<VectorMaskedStoreOpConversion>(typeConverter, context);
    }

    patterns.add<PtrToIntOpConversion>(typeConverter, context);
    patterns.add<IntToPtrOpConversion>(typeConverter, context);
    patterns.add<PtrToMemRefOpConversion>(typeConverter, context);
    patterns.add<AddPtrOpConversion>(typeConverter, context);
    patterns.add<PtrBitcastConversion>(typeConverter, context);
    patterns.add<PtrSelectConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createMemoryOpToLLVMPass() {
  return std::make_unique<MemoryOpToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
