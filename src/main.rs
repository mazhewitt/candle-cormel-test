//! To run this benchmark:
//! 1. Make sure you are on a macOS machine with Xcode installed.
//! 2. Create a new Rust project: `cargo new candle_benchmark && cd candle_benchmark`
//! 3. Add the following dependencies to your `Cargo.toml`:
//!    [dependencies]
//!    candle-core = { version = "0.6.0" }
//!    objc2 = "0.5.2"
//!    objc2-foundation = "0.2.2"
//!    objc2-core-ml = "0.2.0"
//!    rand = "0.8.5"
//!
//! 4. Replace the contents of `src/main.rs` with this code.
//! 5. Run in release mode for accurate timings: `cargo run --release`

use block2::StackBlock;
use candle_core::{Device, Tensor, Error as CandleError, Shape};
use objc2::rc::Retained;
use objc2::AnyThread;
use objc2_core_ml::{MLMultiArray, MLMultiArrayDataType};
use objc2_foundation::{NSArray, NSNumber};
use std::time::Instant;

/// Converts a Candle Tensor to a Core ML MLMultiArray by copying the data.
///
/// This function represents the data marshalling that would happen on every
/// inference call from Candle to a Core ML backend.
///
/// # Arguments
///
/// * `tensor` - A reference to the input `candle_core::Tensor`.
///
/// # Returns
///
/// A `Result` containing the `MLMultiArray` or a `CandleError`.
fn tensor_to_mlmultiarray_by_copy(tensor: &Tensor) -> Result<Retained<MLMultiArray>, CandleError> {
    // Step 1: Get contiguous tensor data - only copy if not already contiguous
    let contiguous_tensor = if tensor.is_contiguous() {
        // No copy needed - use tensor as-is
        tensor.clone()
    } else {
        // Only copy if non-contiguous
        tensor.contiguous()?
    };

    // Step 2: Get element count and data type info
    let element_count = tensor.elem_count();
    let _data_size_bytes = element_count * std::mem::size_of::<f32>();

    // Step 3: Create shape array with pre-allocated capacity
    let dims = tensor.dims();
    let mut shape = Vec::with_capacity(dims.len());
    for &dim in dims {
        shape.push(NSNumber::new_usize(dim));
    }
    let shape_nsarray = NSArray::from_retained_slice(&shape);

    // Step 4: Create MLMultiArray
    let multi_array_result = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape_nsarray,
            MLMultiArrayDataType::Float32
        )
    };
    
    match multi_array_result {
        Ok(ml_array) => {
            // Step 5: Direct memory copy from tensor to MLMultiArray
            use std::sync::atomic::{AtomicBool, Ordering};
            let copied = AtomicBool::new(false);
            
            // Get raw data pointer from contiguous tensor
            let flattened_tensor = contiguous_tensor.flatten_all()?;
            let data_vec = flattened_tensor.to_vec1::<f32>()?;
            
            unsafe {
                ml_array.getMutableBytesWithHandler(&StackBlock::new(|ptr: std::ptr::NonNull<std::ffi::c_void>, len, _| {
                    let dst = ptr.as_ptr() as *mut f32;
                    let src = data_vec.as_ptr();
                    let copy_elements = element_count.min(len as usize / std::mem::size_of::<f32>());
                    
                    // Bounds check
                    if copy_elements > 0 && len as usize >= copy_elements * std::mem::size_of::<f32>() {
                        std::ptr::copy_nonoverlapping(src, dst, copy_elements);
                        copied.store(true, Ordering::Relaxed);
                    }
                }));
            }
            
            if copied.load(Ordering::Relaxed) {
                Ok(ml_array)
            } else {
                Err(CandleError::Msg("Failed to copy data to MLMultiArray".to_string()))
            }
        }
        Err(err) => {
            Err(CandleError::Msg(format!("Failed to create MLMultiArray: {:?}", err)))
        }
    }
}

/// A benchmark runner function.
fn run_benchmark(description: &str, shape: &Shape, device: &Device) {
    println!("--- Running Benchmark: {} ({:?}) ---", description, shape);

    // Create a sample tensor with random data
    let tensor = Tensor::randn(0f32, 1f32, shape, device).unwrap();
    let num_iterations = 1000;
    let mut total_duration = std::time::Duration::new(0, 0);

    // Warm-up run
    let _ = tensor_to_mlmultiarray_by_copy(&tensor).unwrap();

    // Timed runs
    for _ in 0..num_iterations {
        let start_time = Instant::now();
        let _ml_array = tensor_to_mlmultiarray_by_copy(&tensor).unwrap();
        total_duration += start_time.elapsed();
    }

    let avg_duration = total_duration / num_iterations;
    println!(
        "Avg. conversion time over {} iterations: {:?}",
        num_iterations, avg_duration
    );
    println!();
}

fn main() {
    // Use the CPU for this benchmark as we are measuring data marshalling overhead,
    // not GPU computation speed.
    let device = Device::Cpu;
    println!("Using device: {:?}", device);

    // --- Define Benchmark Cases ---

    // 1. Small tensor, like a single embedding vector
    let small_shape = (1, 768).into();
    run_benchmark("Small Tensor (Embedding)", &small_shape, &device);

    // 2. Medium tensor, like an activation in a small LLM
    let medium_shape = (1, 128, 768).into(); // (batch, sequence_len, hidden_dim)
    run_benchmark("Medium Tensor (Activations)", &medium_shape, &device);

    // 3. Large tensor, like a weight matrix in a Transformer block
    let large_shape = (4096, 4096).into(); // (hidden_dim, intermediate_dim)
    run_benchmark("Large Tensor (Weight Matrix)", &large_shape, &device);
    
    // 4. Very Large tensor, simulating a bigger model's weights
    let very_large_shape = (1, 8192, 8192).into();
    run_benchmark("Very Large Tensor", &very_large_shape, &device);
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use objc2_core_ml::MLMultiArrayDataType;

    #[test]
    fn test_tensor_to_mlmultiarray_by_copy() {
        let device = Device::Cpu;
        
        // Create a small test tensor with known values
        let test_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(test_data.clone(), (2, 3), &device).unwrap();
        
        // Convert to MLMultiArray
        let ml_array = tensor_to_mlmultiarray_by_copy(&tensor).unwrap();
        
        // Verify the shape
        let shape = unsafe { ml_array.shape() };
        assert_eq!(shape.count(), 2);
        assert_eq!(shape.objectAtIndex(0).unsignedIntegerValue(), 2);
        assert_eq!(shape.objectAtIndex(1).unsignedIntegerValue(), 3);
        
        // Verify the data type
        assert_eq!(unsafe { ml_array.dataType() }, MLMultiArrayDataType::Float32);
        
        // Verify the data was copied correctly
        use std::cell::RefCell;
        let copied_data = RefCell::new(vec![0.0f32; 6]);
        unsafe {
            ml_array.getBytesWithHandler(&StackBlock::new(|ptr: std::ptr::NonNull<std::ffi::c_void>, len| {
                let src = ptr.as_ptr() as *const f32;
                let mut data = copied_data.borrow_mut();
                let dst = data.as_mut_ptr();
                std::ptr::copy_nonoverlapping(src, dst, (len as usize / std::mem::size_of::<f32>()).min(6));
            }));
        }
        let copied_data = copied_data.into_inner();
        
        // Compare the data
        assert_eq!(copied_data, test_data);
    }
    
    #[test]
    fn test_tensor_to_mlmultiarray_different_shapes() {
        let device = Device::Cpu;
        
        // Test 1D tensor
        let tensor_1d = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let ml_array_1d = tensor_to_mlmultiarray_by_copy(&tensor_1d).unwrap();
        let shape_1d = unsafe { ml_array_1d.shape() };
        assert_eq!(shape_1d.count(), 1);
        assert_eq!(shape_1d.objectAtIndex(0).unsignedIntegerValue(), 3);
        
        // Test 3D tensor
        let tensor_3d = Tensor::from_vec(vec![1.0f32; 24], (2, 3, 4), &device).unwrap();
        let ml_array_3d = tensor_to_mlmultiarray_by_copy(&tensor_3d).unwrap();
        let shape_3d = unsafe { ml_array_3d.shape() };
        assert_eq!(shape_3d.count(), 3);
        assert_eq!(shape_3d.objectAtIndex(0).unsignedIntegerValue(), 2);
        assert_eq!(shape_3d.objectAtIndex(1).unsignedIntegerValue(), 3);
        assert_eq!(shape_3d.objectAtIndex(2).unsignedIntegerValue(), 4);
    }
}
