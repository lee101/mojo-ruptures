"""L2 change-point kernels exposed through a small C ABI."""

from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def l2_cost(sums: FPtr, squares: FPtr, start: Int, end: Int, dims: Int) -> Float64:
    var length = end - start
    var total = 0.0
    for col in range(dims):
        var delta = sums[end * dims + col] - sums[start * dims + col]
        var delta2 = squares[end * dims + col] - squares[start * dims + col]
        var value = delta2 - delta * delta / Float64(length)
        if value > 0.0:
            total += value
    return total


def l2_prefix_plane[square: Bool](
    signal: FPtr,
    destination: FPtr,
    samples: Int,
    dims: Int,
):
    comptime W = simdwidthof[DType.float64]()
    var vector_end = dims - dims % W
    for col in range(0, vector_end, W):
        destination.store(col, SIMD[DType.float64, W](0.0))
    for col in range(vector_end, dims):
        destination[col] = 0.0
    for row in range(samples):
        var row_pos = row * dims
        var next_pos = row_pos + dims
        for col in range(0, vector_end, W):
            var value = (
                signal.load[width=W](row_pos + col)
                - signal.load[width=W](col)
            )
            comptime if square:
                value *= value
            destination.store(
                next_pos + col,
                destination.load[width=W](row_pos + col) + value,
            )
        for col in range(vector_end, dims):
            var value = signal[row_pos + col] - signal[col]
            comptime if square:
                value *= value
            destination[next_pos + col] = destination[row_pos + col] + value


@export("mr_l2_prefix")
def mr_l2_prefix(
    signal_addr: Int,
    sums_addr: Int,
    squares_addr: Int,
    samples: Int,
    dims: Int,
) abi("C"):
    comptime W = simdwidthof[DType.float64]()
    var signal = fp(signal_addr)
    var sums = fp(sums_addr)
    var squares = fp(squares_addr)

    if samples * dims >= 16_000_000:
        l2_prefix_plane[False](signal, sums, samples, dims)
        l2_prefix_plane[True](signal, squares, samples, dims)
        return

    var vector_end = dims - dims % W
    for col in range(0, vector_end, W):
        sums.store(col, SIMD[DType.float64, W](0.0))
        squares.store(col, SIMD[DType.float64, W](0.0))
    for col in range(vector_end, dims):
        sums[col] = 0.0
        squares[col] = 0.0
    for row in range(samples):
        var row_pos = row * dims
        var next_pos = row_pos + dims
        for col in range(0, vector_end, W):
            var value = (
                signal.load[width=W](row_pos + col)
                - signal.load[width=W](col)
            )
            sums.store(
                next_pos + col,
                sums.load[width=W](row_pos + col) + value,
            )
            squares.store(
                next_pos + col,
                squares.load[width=W](row_pos + col) + value * value,
            )
        for col in range(vector_end, dims):
            var value = signal[row_pos + col] - signal[col]
            sums[next_pos + col] = sums[row_pos + col] + value
            squares[next_pos + col] = squares[row_pos + col] + value * value


@export("mr_l2_error")
def mr_l2_error(
    sums_addr: Int,
    squares_addr: Int,
    start: Int,
    end: Int,
    dims: Int,
) abi("C") -> Float64:
    return l2_cost(fp(sums_addr), fp(squares_addr), start, end, dims)


@export("mr_l2_error_many")
def mr_l2_error_many(
    sums_addr: Int,
    squares_addr: Int,
    starts_addr: Int,
    ends_addr: Int,
    result_addr: Int,
    count: Int,
    dims: Int,
) abi("C"):
    var sums = fp(sums_addr)
    var squares = fp(squares_addr)
    var starts = ip(starts_addr)
    var ends = ip(ends_addr)
    var result = fp(result_addr)
    for idx in range(count):
        result[idx] = l2_cost(
            sums, squares, Int(starts[idx]), Int(ends[idx]), dims
        )


@export("mr_dynp")
def mr_dynp(
    sums_addr: Int,
    squares_addr: Int,
    endpoints_addr: Int,
    dp_addr: Int,
    back_addr: Int,
    result_addr: Int,
    endpoint_count: Int,
    dims: Int,
    min_size: Int,
    n_bkps: Int,
) abi("C") -> Int:
    var sums = fp(sums_addr)
    var squares = fp(squares_addr)
    var endpoints = ip(endpoints_addr)
    var dp = fp(dp_addr)
    var back = ip(back_addr)
    var result = ip(result_addr)
    var segments = n_bkps + 1
    var cells = (segments + 1) * endpoint_count
    for idx in range(cells):
        dp[idx] = 1.0e300
        back[idx] = Int64(-1)
    dp[0] = 0.0

    for seg in range(1, segments + 1):
        for end_idx in range(1, endpoint_count):
            var end = Int(endpoints[end_idx])
            var best = 1.0e300
            var best_start = -1
            for start_idx in range(end_idx):
                var start = Int(endpoints[start_idx])
                var previous = dp[(seg - 1) * endpoint_count + start_idx]
                if end - start >= min_size and previous < 1.0e299:
                    var candidate = previous + l2_cost(
                        sums, squares, start, end, dims
                    )
                    if candidate < best:
                        best = candidate
                        best_start = start_idx
            dp[seg * endpoint_count + end_idx] = best
            back[seg * endpoint_count + end_idx] = Int64(best_start)

    var current = endpoint_count - 1
    var seg = segments
    if back[seg * endpoint_count + current] < 0:
        return 0
    while seg > 0:
        result[seg - 1] = endpoints[current]
        current = Int(back[seg * endpoint_count + current])
        seg -= 1
    return segments


@export("mr_pelt")
def mr_pelt(
    sums_addr: Int,
    squares_addr: Int,
    endpoints_addr: Int,
    score_addr: Int,
    back_addr: Int,
    active_addr: Int,
    result_addr: Int,
    endpoint_count: Int,
    dims: Int,
    min_size: Int,
    penalty: Float64,
) abi("C") -> Int:
    var sums = fp(sums_addr)
    var squares = fp(squares_addr)
    var endpoints = ip(endpoints_addr)
    var score = fp(score_addr)
    var back = ip(back_addr)
    var active = ip(active_addr)
    var result = ip(result_addr)
    for idx in range(endpoint_count):
        score[idx] = 1.0e300
        back[idx] = Int64(-1)
        active[idx] = Int64(1)
    score[0] = 0.0

    for end_idx in range(1, endpoint_count):
        var end = Int(endpoints[end_idx])
        var best = 1.0e300
        var best_start = -1
        for start_idx in range(end_idx):
            var start = Int(endpoints[start_idx])
            if (
                active[start_idx] != 0
                and end - start >= min_size
                and score[start_idx] < 1.0e299
            ):
                var candidate = score[start_idx] + l2_cost(
                    sums, squares, start, end, dims
                ) + penalty
                if candidate < best:
                    best = candidate
                    best_start = start_idx
        score[end_idx] = best
        back[end_idx] = Int64(best_start)

        if best_start >= 0:
            for start_idx in range(end_idx):
                var start = Int(endpoints[start_idx])
                if (
                    active[start_idx] != 0
                    and end - start >= min_size
                    and score[start_idx] < 1.0e299
                ):
                    var unpenalized = score[start_idx] + l2_cost(
                        sums, squares, start, end, dims
                    )
                    if unpenalized > best:
                        active[start_idx] = Int64(0)

    var current = endpoint_count - 1
    var count = 0
    while current > 0:
        if back[current] < 0:
            return 0
        result[count] = endpoints[current]
        count += 1
        current = Int(back[current])
    for idx in range(count // 2):
        var other = count - 1 - idx
        var saved = result[idx]
        result[idx] = result[other]
        result[other] = saved
    return count


@export("mr_binseg")
def mr_binseg(
    sums_addr: Int,
    squares_addr: Int,
    result_addr: Int,
    samples: Int,
    dims: Int,
    min_size: Int,
    jump: Int,
    mode: Int,
    target: Float64,
) abi("C") -> Int:
    var sums = fp(sums_addr)
    var squares = fp(squares_addr)
    var result = ip(result_addr)
    result[0] = Int64(samples)
    var count = 1

    while True:
        var best_gain = -1.0e300
        var best_bkp = -1
        var insert_at = -1
        var current_error = 0.0
        var start = 0
        for segment in range(count):
            var end = Int(result[segment])
            var segment_cost = l2_cost(sums, squares, start, end, dims)
            current_error += segment_cost
            var segment_gain = -1.0e300
            var segment_bkp = -1
            var candidate = start
            while candidate < end:
                if (
                    candidate - start >= min_size
                    and end - candidate >= min_size
                ):
                    var gain = segment_cost - l2_cost(
                        sums, squares, start, candidate, dims
                    ) - l2_cost(sums, squares, candidate, end, dims)
                    if gain > segment_gain or (
                        gain == segment_gain and candidate > segment_bkp
                    ):
                        segment_gain = gain
                        segment_bkp = candidate
                candidate += jump
            if segment_bkp >= 0 and segment_gain > best_gain:
                best_gain = segment_gain
                best_bkp = segment_bkp
                insert_at = segment
            start = end

        if best_bkp < 0:
            break
        var should_split: Bool
        if mode == 0:
            should_split = count - 1 < Int(target)
        elif mode == 1:
            should_split = best_gain > target
        else:
            should_split = current_error > target
        if not should_split:
            break

        var move = count
        while move > insert_at:
            result[move] = result[move - 1]
            move -= 1
        result[insert_at] = Int64(best_bkp)
        count += 1
    return count
