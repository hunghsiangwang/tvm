"""
tirx Pass: Transform standard matmul to QuireMatmulElem-based implementation

This pass converts standard matmul patterns (2D, 3D, 4D) to QuireMatmul extern calls.

2D matmul pattern (M=1, K, N):
    for i1 in parallel(N):
        for i0 in unroll(M=1):
            for k in range(K):  # reduction axis
                C[i0, i1] = C[i0, i1] + A[i0, k] * B[k, i1]

    Transforms to:
        for i1 in parallel(N):
            for i0 in unroll(M=1):
                call_extern("Posit{bits}es{es}QuireMatmul", A, offsetA, K, B, offsetB, i1, N, C, offsetC)

3D matmul pattern (batch, M=1, K, N):
    for i2 in parallel(N):
        for i0 in unroll(batch=1):
            for i1, k in grid(M=1, K):  # i1 spatial, k reduction
                C[i0, i1, i2] = C[i0, i1, i2] + A[i0, i1, k] * B[k, i2]

    Transforms to:
        for i2 in parallel(N):
            for i0 in unroll(batch=1):
                for i1 in grid(M=1):
                    call_extern("Posit{bits}es{es}QuireMatmul", A_flat, offsetA, K, B, offsetB, i2, N, C_flat, offsetC)

4D matmul pattern (B, H, M=1, K, N):
    for i3 in parallel(N):
        for i1 in unroll(H):
            for i0, i2, k in grid(B=1, M=1, K):  # k is reduction
                C[i0, i1, i2, i3] = C[i0, i1, i2, i3] + A[i0, i1, i2, k] * B[i0, i1, k, i3]

    Transforms to:
        for i3 in parallel(N):
            for i1 in unroll(H):
                for i0, i2 in grid(B=1, M=1):
                    call_extern("Posit{bits}es{es}QuireMatmul", A_flat, offsetA, K, B_flat, offsetB, i3, N, C_flat, offsetC)
"""

import re

import tvm
from tvm import tirx
from tvm.tirx.functor import PyStmtExprMutator


_POSIT_DTYPE_RE = re.compile(r"^custom\[posites(\d+)\](\d+)(x\d+)?$")


def get_quire_extern_symbol_from_dtype(lhs_dtype, rhs_dtype, output_dtype, with_offset):
    """Return extern symbol name for compatible custom posit buffer dtypes.

    Parameters
    ----------
    lhs_dtype, rhs_dtype, output_dtype : Union[str, tvm.DataType]
        Matmul buffer dtypes to inspect.
    with_offset : bool
        True for the 9-arg function (A/B offsets), False for 7-arg Elem variant.

    Returns
    -------
    Optional[str]
        Symbol name like "Posit12es1QuireMatmulElem" or None if dtype is not custom posit.
    """
    matches = [
        _POSIT_DTYPE_RE.fullmatch(str(dtype))
        for dtype in (lhs_dtype, rhs_dtype, output_dtype)
    ]
    if any(match is None or match.group(3) is not None for match in matches):
        return None

    lhs_es, rhs_es, output_es = (int(match.group(1)) for match in matches)
    lhs_bits, rhs_bits, output_bits = (int(match.group(2)) for match in matches)
    if lhs_es != rhs_es or lhs_es != output_es or lhs_bits != rhs_bits:
        return None

    if not with_offset:
        return None
    if lhs_bits not in (8, 16) or output_bits not in (lhs_bits, 32):
        return None
    return f"Posit{lhs_bits}es{lhs_es}QuireMatmulToPosit{output_bits}"


def transform_matmul_to_quire_elem(func: tirx.PrimFunc) -> tirx.PrimFunc:
    """
    tirx pass that transforms standard matmul loops into QuireMatmul calls.
    Supports 2D, 3D, and 4D matmul patterns.
    """

    @tirx.functor.mutator
    class MatmulTransformer(PyStmtExprMutator):
        def __init__(self):
            super().__init__()
            self.transformed = False

        def visit_for_(self, node):
            """Check for matmul pattern and transform"""
            # Try to match different matmul patterns

            # All matmul patterns start with at least 2 nested For loops
            if not isinstance(node.body, tirx.For):
                return super().visit_for_(node)

            loop1 = node  # Outermost loop
            loop2 = node.body  # Second loop

            if not isinstance(loop2.body, tirx.For):
                return super().visit_for_(node)

            loop3 = loop2.body  # Third loop

            # Pattern 1: 2D matmul (3 nested loops)
            # for i1 (parallel) -> for i0 (unroll) -> for k (serial) -> BlockRealize
            if isinstance(loop3.body, tirx.SBlockRealize):
                block_realize = loop3.body
                block = block_realize.block

                # Check if this is a 2D matmul block (3 iter vars)
                if self.is_matmul_block(block, expected_dims=3):
                    return self.transform_2d_matmul(loop1, loop2, loop3, block_realize, block)

            # Pattern 2 & 3: 3D or 4D matmul (4+ nested loops)
            # for outer -> for middle -> for i1 -> for k -> BlockRealize (3D)
            # for outer -> for middle -> for i0 -> for i2 -> for k -> BlockRealize (4D)
            if isinstance(loop3.body, tirx.For):
                loop4 = loop3.body  # Fourth loop

                # Check for 3D (4 loops total)
                if isinstance(loop4.body, tirx.SBlockRealize):
                    block_realize = loop4.body
                    block = block_realize.block

                    # Check if this is a 3D matmul block (4 iter vars)
                    if self.is_matmul_block(block, expected_dims=4):
                        return self.transform_3d_matmul(loop1, loop2, loop3, loop4, block_realize, block)

                # Check for 4D (5 loops total)
                if isinstance(loop4.body, tirx.For):
                    loop5 = loop4.body  # Fifth loop

                    if isinstance(loop5.body, tirx.SBlockRealize):
                        block_realize = loop5.body
                        block = block_realize.block

                        # Check if this is a 4D matmul block (5 iter vars)
                        if self.is_matmul_block(block, expected_dims=5):
                            return self.transform_4d_matmul(loop1, loop2, loop3, loop4, loop5, block_realize, block)

            # Not a matmul pattern, continue normal traversal
            return super().visit_for_(node)

        def is_matmul_block(self, block, expected_dims):
            """
            Check if block is a matmul reduction.
            expected_dims: 3 for 2D (i0, i1, k), 4 for 3D (i0, i1, i2, k), 5 for 4D (i0, i1, i2, i3, k)
            """
            # Must have expected number of iter vars
            if len(block.iter_vars) != expected_dims:
                return False

            # Must have exactly one reduction axis (the last one)
            reduction_count = sum(
                1 for iv in block.iter_vars
                if iv.iter_type == tirx.IterVar.CommReduce
            )
            if reduction_count != 1:
                return False

            # The last iter var must be the reduction axis
            if block.iter_vars[-1].iter_type != tirx.IterVar.CommReduce:
                return False

            # Check for multiply-add pattern in BufferStore
            if not isinstance(block.body, tirx.BufferStore):
                return False

            store = block.body
            if isinstance(store.value, tirx.Add):
                add = store.value
                product = add.b
                if isinstance(product, tirx.Cast):
                    product = product.value
                if isinstance(product, tirx.Mul):
                    # Additional check: ensure reads contain BufferLoad (not Call nodes)
                    # Matmul blocks should have buffer reads, not extern calls
                    if len(block.reads) >= 2:
                        # Check that we're reading from actual buffers
                        for read_region in block.reads:
                            if not hasattr(read_region, 'buffer'):
                                return False
                        return True

            return False

        @staticmethod
        def _strip_cast(expr):
            while isinstance(expr, tirx.Cast):
                expr = expr.value
            return expr

        def transform_matmul(self, spatial_loops, k_loop, block_realize, block):
            """Replace one reduction with a layout-derived quire extern call.

            Loop order is not a reliable indication of tensor dimensions after
            TIR scheduling.  Derive all flattened offsets from the actual
            BufferLoad/BufferStore indices instead.
            """

            store = block.body
            product = self._strip_cast(store.value.b)
            if not isinstance(product, tirx.Mul):
                return spatial_loops[0]
            lhs_load = self._strip_cast(product.a)
            rhs_load = self._strip_cast(product.b)
            if not isinstance(lhs_load, tirx.BufferLoad) or not isinstance(
                rhs_load, tirx.BufferLoad
            ):
                return spatial_loops[0]

            C_buffer = store.buffer
            lhs_buffer = lhs_load.buffer
            rhs_buffer = rhs_load.buffer
            extern_symbol = get_quire_extern_symbol_from_dtype(
                lhs_buffer.dtype,
                rhs_buffer.dtype,
                C_buffer.dtype,
                with_offset=True,
            )
            if extern_symbol is None:
                return spatial_loops[0]

            block_bindings = {
                iter_var.var: value
                for iter_var, value in zip(block.iter_vars, block_realize.iter_values)
            }

            def bind_indices(indices):
                return [
                    tirx.stmt_functor.substitute(index, block_bindings)
                    for index in indices
                ]

            lhs_indices = bind_indices(lhs_load.indices)
            rhs_indices = bind_indices(rhs_load.indices)
            output_indices = bind_indices(store.indices)
            reduction_var = k_loop.loop_var
            reduction_min = k_loop.min
            next_reduction = reduction_min + 1

            def flat_offset(buffer, indices):
                if buffer.ty.strides:
                    offset = buffer.ty.elem_offset
                    for index, stride in zip(indices, buffer.ty.strides):
                        offset = offset + index * stride
                    return offset
                offset = 0
                for index, extent in zip(indices, buffer.ty.shape):
                    offset = offset * extent + index
                return buffer.ty.elem_offset + offset

            def offset_at(buffer, indices, reduction_value):
                bound = [
                    tirx.stmt_functor.substitute(
                        index, {reduction_var: reduction_value}
                    )
                    for index in indices
                ]
                return flat_offset(buffer, bound)

            analyzer = tvm.arith.Analyzer()
            lhs_base = analyzer.simplify(
                offset_at(lhs_buffer, lhs_indices, reduction_min)
            )
            rhs_base = analyzer.simplify(
                offset_at(rhs_buffer, rhs_indices, reduction_min)
            )
            lhs_stride = analyzer.simplify(
                offset_at(lhs_buffer, lhs_indices, next_reduction) - lhs_base
            )
            rhs_stride = analyzer.simplify(
                offset_at(rhs_buffer, rhs_indices, next_reduction) - rhs_base
            )
            # The C++ helper walks its first input contiguously.  Scalar
            # multiplication is commutative, so swap operands when only the
            # right-hand reduction dimension is contiguous.
            if not analyzer.can_prove_equal(lhs_stride, 1):
                if not analyzer.can_prove_equal(rhs_stride, 1):
                    return spatial_loops[0]
                lhs_buffer, rhs_buffer = rhs_buffer, lhs_buffer
                lhs_base, rhs_base = rhs_base, lhs_base
                lhs_stride, rhs_stride = rhs_stride, lhs_stride

            output_offset = analyzer.simplify(flat_offset(C_buffer, output_indices))
            extern_call = tirx.Evaluate(
                tirx.call_extern(
                    "int32",
                    extern_symbol,
                    lhs_buffer.data,
                    lhs_base,
                    k_loop.extent,
                    rhs_buffer.data,
                    rhs_base,
                    tirx.const(0, "int64"),
                    rhs_stride,
                    C_buffer.data,
                    output_offset,
                )
            )

            body = extern_call
            for loop in reversed(spatial_loops):
                body = tirx.For(
                    loop.loop_var,
                    loop.min,
                    loop.extent,
                    loop.kind,
                    body,
                    loop.thread_binding,
                    loop.annotations,
                )
            self.transformed = True
            return body

        def transform_2d_matmul(self, i1_loop, i0_loop, k_loop, block_realize, block):
            """Transform 2D matmul loops to QuireMatmulElem call"""
            return self.transform_matmul(
                [i1_loop, i0_loop], k_loop, block_realize, block
            )

        def transform_3d_matmul(self, i2_loop, i0_loop, i1_loop, k_loop, block_realize, block):
            """
            Transform 3D matmul: (batch, M, K) @ (K, N) -> (batch, M, N)
            Pattern: for i2(N) -> for i0(batch) -> for i1(M) -> for k(K) -> block

            Transform to:
                for i2(N) -> for i0(batch) -> for i1(M) -> QuireMatmulElemWithOffset call
            """
            return self.transform_matmul(
                [i2_loop, i0_loop, i1_loop], k_loop, block_realize, block
            )

        def transform_4d_matmul(self, i3_loop, i1_loop, i0_loop, i2_loop, k_loop, block_realize, block):
            """
            Transform 4D matmul: (B, H, M, K) @ (B, H, K, N) -> (B, H, M, N)
            Pattern: for i3(N) -> for i1(H) -> for i0(B) -> for i2(M) -> for k(K) -> block

            Transform to:
                for i3(N) -> for i1(H) -> for i0(B) -> for i2(M) -> QuireMatmulElemWithOffset call
            """
            return self.transform_matmul(
                [i3_loop, i1_loop, i0_loop, i2_loop],
                k_loop,
                block_realize,
                block,
            )

    # Apply transformation
    transformer = MatmulTransformer()

    # Handle root block if present
    body = func.body
    if isinstance(body, tirx.SBlockRealize) and body.block.name_hint == "root":
        # Visit the body of the root block
        new_body = transformer.visit_stmt(body.block.body)
        # Reconstruct root block with transformed body
        new_root_block = tirx.SBlock(
            iter_vars=body.block.iter_vars,
            reads=body.block.reads,
            writes=body.block.writes,
            name_hint=body.block.name_hint,
            body=new_body,
            init=body.block.init,
            alloc_buffers=body.block.alloc_buffers,
            match_buffers=body.block.match_buffers,
            annotations=body.block.annotations
        )
        new_body = tirx.SBlockRealize(
            iter_values=body.iter_values,
            predicate=body.predicate,
            block=new_root_block
        )
    else:
        new_body = transformer.visit_stmt(body)

    if transformer.transformed:
        # Create new function with transformed body
        new_func = func.with_body(new_body)
        # Add tirx.noalias attribute
        return new_func.with_attr("tirx.noalias", True)
    else:
        return func


@tvm.tirx.transform.prim_func_pass(opt_level=0)
class InjectQuireMatmulElem:
    """
    TVM pass to transform matmul operations to use QuireMatmulElem extern calls

    Usage:
        mod = IRModule({"matmul": matmul_func})
        transformed_mod = InjectQuireMatmulElem()(mod)
    """

    def __init__(self):
        pass

    def transform_function(self, func, mod, ctx):
        return transform_matmul_to_quire_elem(func)
