; ModuleID = 'shfl_up.cl'
source_filename = "shfl_up.cl"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@shfl_up_sync_updated = dso_local local_unnamed_addr global [32 x i8] zeroinitializer, align 16
@shfl_up_sync_shared_var_arr = common dso_local local_unnamed_addr global [32 x i32] zeroinitializer, align 16

; Function Attrs: convergent nofree nounwind uwtable
define dso_local i32 @shfl_up_sync(i32 %0, i32 %1, i32 %2, i32 %3) local_unnamed_addr #0 {
  %5 = tail call i64 @_Z13get_global_idj(i32 0) #2
  %6 = trunc i64 %5 to i32
  %7 = shl i64 %5, 32
  %8 = ashr exact i64 %7, 32
  %9 = getelementptr inbounds [32 x i32], [32 x i32]* @shfl_up_sync_shared_var_arr, i64 0, i64 %8
  store i32 %1, i32* %9, align 4, !tbaa !3
  %10 = getelementptr inbounds [32 x i8], [32 x i8]* @shfl_up_sync_updated, i64 0, i64 %8
  store i8 1, i8* %10, align 1, !tbaa !7
  %11 = and i32 %6, 31
  %12 = shl nuw i32 1, %11
  %13 = and i32 %12, %0
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %35, label %15

15:                                               ; preds = %4
  %16 = sitofp i32 %6 to float
  %17 = sitofp i32 %3 to float
  %18 = fdiv float %16, %17, !fpmath !9
  %19 = tail call float @_Z5floorf(float %18) #2
  %20 = srem i32 %6, %3
  %21 = sub i32 %20, %2
  %22 = icmp sgt i32 %21, -1
  br i1 %22, label %23, label %35

23:                                               ; preds = %15
  %24 = fptosi float %19 to i32
  %25 = mul nsw i32 %24, %3
  %26 = add nsw i32 %21, %25
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds [32 x i8], [32 x i8]* @shfl_up_sync_updated, i64 0, i64 %27
  %29 = load i8, i8* %28, align 1, !tbaa !7, !range !10
  %30 = icmp eq i8 %29, 0
  br label %31

31:                                               ; preds = %23, %31
  br i1 %30, label %31, label %32

32:                                               ; preds = %31
  %33 = getelementptr inbounds [32 x i32], [32 x i32]* @shfl_up_sync_shared_var_arr, i64 0, i64 %27
  %34 = load i32, i32* %33, align 4, !tbaa !3
  store i8 0, i8* %28, align 1, !tbaa !7
  br label %35

35:                                               ; preds = %15, %32, %4
  %36 = phi i32 [ %1, %4 ], [ %34, %32 ], [ %1, %15 ]
  ret i32 %36
}

; Function Attrs: convergent nounwind readnone
declare dso_local i64 @_Z13get_global_idj(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local float @_Z5floorf(float) local_unnamed_addr #1

; Function Attrs: convergent nofree nounwind uwtable
define dso_local void @test_shfl_up_sync_custom() local_unnamed_addr #0 {
  %1 = tail call i64 @_Z13get_global_idj(i32 0) #2
  %2 = trunc i64 %1 to i32
  %3 = and i32 %2, 31
  %4 = xor i32 %3, 31
  %5 = shl i64 %1, 32
  %6 = ashr exact i64 %5, 32
  %7 = getelementptr inbounds [32 x i32], [32 x i32]* @shfl_up_sync_shared_var_arr, i64 0, i64 %6
  %8 = getelementptr inbounds [32 x i8], [32 x i8]* @shfl_up_sync_updated, i64 0, i64 %6
  %9 = sitofp i32 %2 to float
  %10 = fmul float %9, 1.250000e-01
  %11 = tail call float @_Z5floorf(float %10) #2
  %12 = srem i32 %2, 8
  %13 = fptosi float %11 to i32
  %14 = shl i32 %13, 3
  %15 = and i32 %2, 7
  store i32 %4, i32* %7, align 4, !tbaa !3
  store i8 1, i8* %8, align 1, !tbaa !7
  %16 = icmp sgt i32 %12, 0
  br i1 %16, label %21, label %17

17:                                               ; preds = %0
  %18 = icmp eq i32 %15, 0
  %19 = select i1 %18, i32 0, i32 %4
  %20 = add nuw nsw i32 %19, %4
  store i32 %20, i32* %7, align 4, !tbaa !3
  store i8 1, i8* %8, align 1, !tbaa !7
  br label %44

21:                                               ; preds = %0
  %22 = add nsw i32 %12, -1
  %23 = add nsw i32 %14, %22
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds [32 x i8], [32 x i8]* @shfl_up_sync_updated, i64 0, i64 %24
  %26 = load i8, i8* %25, align 1, !tbaa !7, !range !10
  %27 = icmp eq i8 %26, 0
  br i1 %27, label %28, label %30

28:                                               ; preds = %56, %37, %21
  br label %29

29:                                               ; preds = %28, %29
  br label %29

30:                                               ; preds = %21
  %31 = getelementptr inbounds [32 x i32], [32 x i32]* @shfl_up_sync_shared_var_arr, i64 0, i64 %24
  %32 = load i32, i32* %31, align 4, !tbaa !3
  store i8 0, i8* %25, align 1, !tbaa !7
  %33 = icmp eq i32 %15, 0
  %34 = select i1 %33, i32 0, i32 %32
  %35 = add i32 %34, %4
  store i32 %35, i32* %7, align 4, !tbaa !3
  store i8 1, i8* %8, align 1, !tbaa !7
  %36 = icmp sgt i32 %12, 1
  br i1 %36, label %37, label %44

37:                                               ; preds = %30
  %38 = add nsw i32 %12, -2
  %39 = add nsw i32 %14, %38
  %40 = sext i32 %39 to i64
  %41 = getelementptr inbounds [32 x i8], [32 x i8]* @shfl_up_sync_updated, i64 0, i64 %40
  %42 = load i8, i8* %41, align 1, !tbaa !7, !range !10
  %43 = icmp eq i8 %42, 0
  br i1 %43, label %28, label %49

44:                                               ; preds = %17, %30
  %45 = phi i32 [ %20, %17 ], [ %35, %30 ]
  %46 = icmp ult i32 %15, 2
  %47 = select i1 %46, i32 0, i32 %45
  %48 = add i32 %47, %45
  store i32 %48, i32* %7, align 4, !tbaa !3
  store i8 1, i8* %8, align 1, !tbaa !7
  br label %64

49:                                               ; preds = %37
  %50 = getelementptr inbounds [32 x i32], [32 x i32]* @shfl_up_sync_shared_var_arr, i64 0, i64 %40
  %51 = load i32, i32* %50, align 4, !tbaa !3
  store i8 0, i8* %41, align 1, !tbaa !7
  %52 = icmp ult i32 %15, 2
  %53 = select i1 %52, i32 0, i32 %51
  %54 = add i32 %53, %35
  store i32 %54, i32* %7, align 4, !tbaa !3
  store i8 1, i8* %8, align 1, !tbaa !7
  %55 = icmp sgt i32 %12, 3
  br i1 %55, label %56, label %64

56:                                               ; preds = %49
  %57 = add nsw i32 %12, -4
  %58 = add nsw i32 %14, %57
  %59 = sext i32 %58 to i64
  %60 = getelementptr inbounds [32 x i8], [32 x i8]* @shfl_up_sync_updated, i64 0, i64 %59
  %61 = load i8, i8* %60, align 1, !tbaa !7, !range !10
  %62 = icmp eq i8 %61, 0
  br i1 %62, label %28, label %63

63:                                               ; preds = %56
  store i8 0, i8* %60, align 1, !tbaa !7
  br label %64

64:                                               ; preds = %44, %63, %49
  ret void
}

attributes #0 = { convergent nofree nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind readnone }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 10.0.0-4ubuntu1 "}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"bool", !5, i64 0}
!9 = !{float 2.500000e+00}
!10 = !{i8 0, i8 2}
