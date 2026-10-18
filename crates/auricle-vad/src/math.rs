//! Shared inner loops for both detectors.

/// Dot product with two f32x4 accumulators on wasm32 (two lanes of
/// independent adds keep the multiply-add pipeline busy).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;
    let n = a.len().min(b.len());
    let (mut acc0, mut acc1) = (f32x4_splat(0.0), f32x4_splat(0.0));
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: i + 8 <= n keeps every load inside both slices.
        unsafe {
            let (pa, pb) = (a.as_ptr().add(i), b.as_ptr().add(i));
            acc0 = f32x4_add(acc0, f32x4_mul(v128_load(pa as *const v128), v128_load(pb as *const v128)));
            acc1 = f32x4_add(acc1, f32x4_mul(v128_load(pa.add(4) as *const v128), v128_load(pb.add(4) as *const v128)));
        }
        i += 8;
    }
    let acc = f32x4_add(acc0, acc1);
    let mut total = f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc) + f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc);
    while i < n {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

/// Dot product with eight independent accumulators so LLVM can emit SIMD lanes
/// natively (a plain `sum()` must keep float order and stays scalar).
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0.0f32; 8];
    let mut i = 0;
    while i + 8 <= n {
        for l in 0..8 {
            acc[l] += a[i + l] * b[i + l];
        }
        i += 8;
    }
    let mut total = acc.iter().sum::<f32>();
    while i < n {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reinterprets an `include_bytes!` blob as little-endian f32 without copying
/// when the linker aligned it, with a leaked copy otherwise, so it is never UB.
pub(crate) fn f32_slice(bytes: &'static [u8]) -> &'static [f32] {
    let (head, body, tail) = unsafe { bytes.align_to::<f32>() };
    if head.is_empty() && tail.is_empty() && cfg!(target_endian = "little") {
        return body;
    }
    let copied: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    Box::leak(copied.into_boxed_slice())
}

/// In-place iterative radix-2 FFT of length `N` (a power of two) on split
/// real and imaginary buffers. `twiddle` holds cos and -sin of 2*pi*k/N for k < N/2.
pub(crate) fn fft<const N: usize>(re: &mut [f32; N], im: &mut [f32; N], twiddle_re: &[f32], twiddle_im: &[f32]) {
    let mut len = 2;
    while len <= N {
        let step = N / len;
        for start in (0..N).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = (twiddle_re[k * step], twiddle_im[k * step]);
                let (a, b) = (start + k, start + k + len / 2);
                let xr = re[b] * wr - im[b] * wi;
                let xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len *= 2;
    }
}

/// Index of `i` with its `bits` low bits reversed (the FFT input order).
#[inline]
pub(crate) fn bit_reverse(i: usize, bits: u32) -> usize {
    (i as u32).reverse_bits() as usize >> (32 - bits)
}

pub(crate) fn twiddles(n: usize) -> (Vec<f32>, Vec<f32>) {
    (0..n / 2)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            (angle.cos() as f32, -(angle.sin() as f32))
        })
        .unzip()
}
