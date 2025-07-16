//! Real CoreML Integration Tests
//!
//! This module contains comprehensive tests for CoreML integration with Candle.
//! Tests will FAIL when features are unimplemented - no mocks, no fallbacks.
//!
//! Usage:
//! cargo test --bin coreml_test -- --test-threads=1
//! cargo run --bin coreml_test -- models/OpenELM.mlmodelc

use block2::StackBlock;
use candle_core::{Device, Tensor, Error as CandleError};
use objc2::rc::{autoreleasepool, Retained};
use objc2::AnyThread;
use objc2_core_ml::{
    MLModel, MLMultiArray, MLMultiArrayDataType, MLFeatureValue, 
    MLDictionaryFeatureProvider, MLFeatureProvider
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};
use objc2::runtime::ProtocolObject;
use std::env;
use std::path::Path;
use tokenizers::Tokenizer;

// Test configuration - OpenELM-450M-Instruct model specifications
const MODEL_PATH: &str = "OpenELM-450M-Instruct-128-float32.mlmodelc";
const TOKENIZER_PATH: &str = "tokenizer.json";
const INPUT_NAME: &str = "input_ids";
const OUTPUT_NAME: &str = "logits";
const MAX_SEQUENCE_LENGTH: usize = 128;  // Fixed sequence length
const VOCAB_SIZE: usize = 32_000;        // LLaMA-2 vocabulary size
const PAD_TOKEN_ID: u32 = 0;             // Padding token

/// Converts a Candle Tensor to a Core ML MLMultiArray (optimized version)
fn tensor_to_mlmultiarray(tensor: &Tensor) -> Result<Retained<MLMultiArray>, CandleError> {
    let contiguous_tensor = if tensor.is_contiguous() {
        tensor.clone()
    } else {
        tensor.contiguous()?
    };

    let element_count = tensor.elem_count();
    let dims = tensor.dims();
    let mut shape = Vec::with_capacity(dims.len());
    for &dim in dims {
        shape.push(NSNumber::new_usize(dim));
    }
    let shape_nsarray = NSArray::from_retained_slice(&shape);

    let multi_array_result = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape_nsarray,
            MLMultiArrayDataType::Float32
        )
    };
    
    match multi_array_result {
        Ok(ml_array) => {
            use std::sync::atomic::{AtomicBool, Ordering};
            let copied = AtomicBool::new(false);
            
            let flattened_tensor = contiguous_tensor.flatten_all()?;
            let data_vec = flattened_tensor.to_vec1::<f32>()?;
            
            unsafe {
                ml_array.getMutableBytesWithHandler(&StackBlock::new(|ptr: std::ptr::NonNull<std::ffi::c_void>, len, _| {
                    let dst = ptr.as_ptr() as *mut f32;
                    let src = data_vec.as_ptr();
                    let copy_elements = element_count.min(len as usize / std::mem::size_of::<f32>());
                    
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

/// Creates a proper MLDictionaryFeatureProvider for model input
fn create_feature_provider(
    input_name: &str,
    input_array: &MLMultiArray,
) -> Result<Retained<MLDictionaryFeatureProvider>, String> {
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::{ns_string, NSDictionary, NSString};
    use objc2_core_ml::{MLDictionaryFeatureProvider, MLFeatureValue};
    use objc2::runtime::AnyObject;

    objc2::rc::autoreleasepool(|_| {
        // Key and value
        let key = NSString::from_str(input_name);          // Retained<NSString>
        let value = unsafe { MLFeatureValue::featureValueWithMultiArray(input_array) };

        // Build single-pair dictionary
        let dict: Retained<NSDictionary<NSString, AnyObject>> =
            NSDictionary::from_slices::<NSString>(&[&*key], &[&*value]);

        // Create the provider
        unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                dict.as_ref(),
            )
        }
        .map_err(|e| format!("CoreML initWithDictionary_error: {:?}", e))
    })
}

/// Runs actual model prediction - NO MOCKS
fn run_model_prediction(
    model: &MLModel,
    provider: &MLDictionaryFeatureProvider,
) -> Result<Retained<ProtocolObject<dyn MLFeatureProvider>>, String> {
    use objc2::runtime::ProtocolObject;
    use objc2_core_ml::MLFeatureProvider;

    objc2::rc::autoreleasepool(|_| unsafe {
        // Convert MLDictionaryFeatureProvider to ProtocolObject
        let protocol_provider = ProtocolObject::from_ref(provider);
        
        model
            .predictionFromFeatures_error(protocol_provider)
            .map_err(|e| format!("CoreML prediction error: {:?}", e))
    })
}

/// Extracts logits from model output - NO MOCKS
fn extract_logits(
    prediction: &ProtocolObject<dyn MLFeatureProvider>,
    output_name: &str,
) -> Result<Vec<f32>, String> {
    objc2::rc::autoreleasepool(|_| unsafe {
        let name = NSString::from_str(output_name);
        let value = prediction
            .featureValueForName(&name)
            .ok_or_else(|| format!("Output '{}' not found", output_name))?;

        let marray = value
            .multiArrayValue()
            .ok_or_else(|| format!("Output '{}' is not MLMultiArray", output_name))?;

        let count = marray.count() as usize;
        let mut buf = vec![0.0f32; count];
        
        // Use a cell pattern to allow mutation in the Fn closure
        use std::cell::RefCell;
        let buf_cell = RefCell::new(&mut buf);
        
        marray.getBytesWithHandler(&block2::StackBlock::new(
            |ptr: std::ptr::NonNull<std::ffi::c_void>, len: isize| {
                let src = ptr.as_ptr() as *const f32;
                let copy_elements = count.min(len as usize / std::mem::size_of::<f32>());
                if copy_elements > 0 && len as usize >= copy_elements * std::mem::size_of::<f32>() {
                    unsafe {
                        if let Ok(mut buf_ref) = buf_cell.try_borrow_mut() {
                            std::ptr::copy_nonoverlapping(src, buf_ref.as_mut_ptr(), copy_elements);
                        }
                    }
                }
            },
        ));

        Ok(buf)
    })
}

/// Tokenizes text and prepares it for OpenELM model (pad/truncate to 128, convert to f32)
fn prepare_model_input(text: &str, tokenizer: &Tokenizer, device: &Device) -> Result<Tensor, String> {
    // 1. Tokenize the text
    let encoding = tokenizer.encode(text, true).map_err(|e| format!("Tokenization failed: {}", e))?;
    let mut tokens = encoding.get_ids().to_vec();
    
    // 2. Truncate to max length (keep last tokens if too long)
    tokens.truncate(MAX_SEQUENCE_LENGTH);
    
    // 3. Pad to exactly 128 tokens with pad_token_id = 0
    tokens.resize(MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID);
    
    // 4. Convert to f32 and create tensor (1, 128)
    let token_f32: Vec<f32> = tokens.into_iter().map(|id| id as f32).collect();
    Tensor::from_vec(token_f32, (1, MAX_SEQUENCE_LENGTH), device)
        .map_err(|e| format!("Tensor creation failed: {}", e))
}

/// Finds the token with highest probability
fn argmax(logits: &[f32]) -> usize {
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx = 0;
    for (i, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

/// Load model - NO FALLBACKS
fn load_model(path: &Path) -> Result<Retained<MLModel>, String> {
    if !path.exists() {
        return Err(format!("Model file not found: {}", path.display()));
    }
    
    autoreleasepool(|_| {
        let url = unsafe { NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy())) };
        match unsafe { MLModel::modelWithContentsOfURL_error(&url) } {
            Ok(model) => Ok(model),
            Err(err) => Err(format!("Failed to load CoreML model: {:?}", err))
        }
    })
}

/// Load tokenizer - NO FALLBACKS
fn load_tokenizer(path: &Path) -> Result<Tokenizer, String> {
    if !path.exists() {
        return Err(format!("Tokenizer file not found: {}", path.display()));
    }
    
    Tokenizer::from_file(path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_tensor_to_mlmultiarray_conversion() {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();
        
        let ml_array = tensor_to_mlmultiarray(&tensor).unwrap();
        
        // Verify shape
        let shape = unsafe { ml_array.shape() };
        assert_eq!(shape.count(), 2);
        assert_eq!(shape.objectAtIndex(0).unsignedIntegerValue(), 2);
        assert_eq!(shape.objectAtIndex(1).unsignedIntegerValue(), 2);
        
        // Verify data type
        assert_eq!(unsafe { ml_array.dataType() }, MLMultiArrayDataType::Float32);
    }

    #[test]
    fn test_model_loading() {
        let model_path = Path::new(MODEL_PATH);
        let model = load_model(model_path).expect("Failed to load model");
        
        // Verify model has expected inputs and outputs
        let model_description = unsafe { model.modelDescription() };
        let input_descriptions = unsafe { model_description.inputDescriptionsByName() };
        let output_descriptions = unsafe { model_description.outputDescriptionsByName() };
        
        assert_eq!(input_descriptions.count(), 1);
        assert_eq!(output_descriptions.count(), 1);
        
        // Verify input/output names
        let input_keys = input_descriptions.allKeys();
        let output_keys = output_descriptions.allKeys();
        
        // Check input name matches expected
        autoreleasepool(|pool| {
            let input_key = input_keys.objectAtIndex(0);
            let input_name = unsafe { input_key.to_str(pool) };
            assert_eq!(input_name, INPUT_NAME);
        });
        
        // Check output name matches expected
        autoreleasepool(|pool| {
            let output_key = output_keys.objectAtIndex(0);
            let output_name = unsafe { output_key.to_str(pool) };
            assert_eq!(output_name, OUTPUT_NAME);
        });
    }

    #[test]
    fn test_tokenizer_loading() {
        let tokenizer_path = Path::new(TOKENIZER_PATH);
        let _tokenizer = load_tokenizer(tokenizer_path).expect("Failed to load tokenizer");
        // If we get here, tokenizer loaded successfully
    }

    #[test]
    fn test_tokenization() {
        let tokenizer_path = Path::new(TOKENIZER_PATH);
        let tokenizer = load_tokenizer(tokenizer_path).expect("Failed to load tokenizer");

        let prompt = "The quick brown fox jumps over the lazy";
        let encoding = tokenizer.encode(prompt, true).expect("Failed to encode prompt");
        let tokens = encoding.get_ids();

        // Real OpenELM tokenizer produces 11 tokens for this prompt
        assert_eq!(tokens.len(), 11, "Tokenizer should yield 11 tokens");
        assert_eq!(
            tokens[..4],
            [1, 450, 4996, 17354], // Start token, "The", " quick", " brown"
            "First four tokens should match known IDs"
        );
    }

    #[test]
    fn test_feature_provider_creation() {
        let device = Device::Cpu;
        // Use correct OpenELM input shape (1, 128) with f32 token IDs
        let token_ids: Vec<f32> = (0..MAX_SEQUENCE_LENGTH).map(|i| (i % 1000) as f32).collect();
        let tensor = Tensor::from_vec(token_ids, (1, MAX_SEQUENCE_LENGTH), &device).unwrap();
        let ml_array = tensor_to_mlmultiarray(&tensor).unwrap();
        
        let provider = create_feature_provider(INPUT_NAME, &ml_array)
            .expect("Failed to create feature provider");
        
        // Feature provider should be created successfully - just verify it exists
        // (No need to check null since Retained<> guarantees non-null)
    }

    #[test] 
    #[ignore] // Ready for real model - remove ignore to test
    fn test_model_prediction() {
        let model_path = Path::new(MODEL_PATH);
        let tokenizer_path = Path::new(TOKENIZER_PATH);
        
        let model = load_model(model_path).expect("Failed to load model");
        let tokenizer = load_tokenizer(tokenizer_path).expect("Failed to load tokenizer");
        
        let device = Device::Cpu;
        let prompt = "The quick brown fox";
        let tensor = prepare_model_input(prompt, &tokenizer, &device)
            .expect("Failed to prepare model input");
        let ml_array = tensor_to_mlmultiarray(&tensor).unwrap();
        let provider = create_feature_provider(INPUT_NAME, &ml_array)
            .expect("Failed to create feature provider");
        
        let _prediction = run_model_prediction(&model, &provider)
            .expect("Failed to run model prediction");
    }

    #[test]
    #[ignore] // Ready for real model - remove ignore to test
    fn test_logits_extraction() {
        let model_path = Path::new(MODEL_PATH);
        let tokenizer_path = Path::new(TOKENIZER_PATH);
        
        let model = load_model(model_path).expect("Failed to load model");
        let tokenizer = load_tokenizer(tokenizer_path).expect("Failed to load tokenizer");
        
        let device = Device::Cpu;
        let prompt = "The quick brown fox";
        let tensor = prepare_model_input(prompt, &tokenizer, &device)
            .expect("Failed to prepare model input");
        let ml_array = tensor_to_mlmultiarray(&tensor).unwrap();
        let provider = create_feature_provider(INPUT_NAME, &ml_array)
            .expect("Failed to create feature provider");
        
        let prediction = run_model_prediction(&model, &provider)
            .expect("Failed to run model prediction");
        
        let logits = extract_logits(&*prediction, OUTPUT_NAME)
            .expect("Failed to extract logits");
        
        // Verify output shape: should be (1 * 128 * 32000) = 4,096,000 logits
        let expected_size = 1 * MAX_SEQUENCE_LENGTH * VOCAB_SIZE;
        assert_eq!(logits.len(), expected_size, "Logits should have shape (1, 128, 32000)");
    }

    #[test]
    fn test_quick_brown_fox_prediction() {
        let model_path = Path::new(MODEL_PATH);
        let tokenizer_path = Path::new(TOKENIZER_PATH);
        
        let model = load_model(model_path).expect("Failed to load model");
        let tokenizer = load_tokenizer(tokenizer_path).expect("Failed to load tokenizer");
        
        let device = Device::Cpu;
        let prompt = "The quick brown fox jumps over the lazy";
        
        // Get original tokens to find actual length
        let encoding = tokenizer.encode(prompt, true).expect("Failed to encode prompt");
        let original_tokens = encoding.get_ids().to_vec();
        let actual_len = original_tokens.len().min(MAX_SEQUENCE_LENGTH);
        
        // Use proper input preparation with padding/truncation to 128
        let tensor = prepare_model_input(prompt, &tokenizer, &device)
            .expect("Failed to prepare model input");
        
        // Convert to MLMultiArray
        let ml_array = tensor_to_mlmultiarray(&tensor).unwrap();
        
        // Create feature provider
        let provider = create_feature_provider(INPUT_NAME, &ml_array)
            .expect("Failed to create feature provider");
        
        // Run prediction
        let prediction = run_model_prediction(&model, &provider)
            .expect("Failed to run model prediction");
        
        // Extract logits - shape should be (1, 128, 32000) flattened
        let logits = extract_logits(&*prediction, OUTPUT_NAME)
            .expect("Failed to extract logits");
        
        // For next-token prediction: use only the last non-pad position's logits
        // Use actual token length, not the padded position
        let last_pos = actual_len - 1;
        let last_position_start = last_pos * VOCAB_SIZE;
        let last_token_logits = &logits[last_position_start..last_position_start + VOCAB_SIZE];
        
        // Find most likely next token
        let next_token_id = argmax(last_token_logits) as u32;
        
        println!("Prompt: '{}'", prompt);
        println!("Actual token length: {}, using position: {}", actual_len, last_pos);
        println!("Predicted token ID: {}", next_token_id);
        println!("Top 5 logits at position {}:", last_pos);
        let mut indexed_logits: Vec<(usize, f32)> = last_token_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, (token_id, logit_value)) in indexed_logits.iter().take(5).enumerate() {
            println!("  {}: token_id={}, logit={:.4}", i+1, token_id, logit_value);
        }
        
        // Decode next token
        let next_token_str = tokenizer.decode(&[next_token_id], true)
            .expect("Failed to decode next token");
        
        println!("Next token predicted: '{}'", next_token_str.trim());
        
        // Log the completion for verification 
        if !next_token_str.trim().is_empty() {
            let completion = format!("{} {}", prompt, next_token_str.trim());
            println!("Full completion: '{}'", completion);
        } else {
            println!("Model predicted token ID 0 (pad token) or empty string");
        }
        
        // FAIL if we don't get "dog" - any language model should predict this!
        let predicted_word = next_token_str.trim().to_lowercase();
        assert_eq!(predicted_word, "dog", 
            "Model should predict 'dog' after 'The quick brown fox jumps over the lazy' but got: '{}'", 
            predicted_word);
    }

    #[test]
    fn test_argmax_functionality() {
        let logits = vec![0.1, 0.9, 0.3, 0.7, 0.2];
        let max_idx = argmax(&logits);
        assert_eq!(max_idx, 1); // Index of 0.9
    }

    #[test]
    #[ignore] // Ready for real model - remove ignore to test
    fn test_extract_logits_unit() {
        let model_path = Path::new(MODEL_PATH);
        let tokenizer_path = Path::new(TOKENIZER_PATH);
        
        let model = load_model(model_path).unwrap();
        let tokenizer = load_tokenizer(tokenizer_path).unwrap();
        
        let device = Device::Cpu;
        let prompt = "Hello world";
        let tensor = prepare_model_input(prompt, &tokenizer, &device).unwrap();
        let ml_array = tensor_to_mlmultiarray(&tensor).unwrap();

        let provider = create_feature_provider(INPUT_NAME, &ml_array).unwrap();
        let pred = run_model_prediction(&model, &provider).unwrap();
        let logits = extract_logits(&*pred, OUTPUT_NAME).unwrap();

        // Should extract exactly (1 * 128 * 32000) logits
        let expected_size = 1 * MAX_SEQUENCE_LENGTH * VOCAB_SIZE;
        assert_eq!(logits.len(), expected_size);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    println!("CoreML Integration Test Runner");
    println!("==============================");
    
    if args.len() < 2 {
        println!("Usage: {} <model_path.mlmodelc>", args[0]);
        println!("Example: cargo run --bin coreml_test -- OpenELM-450M-Instruct-128-float32.mlmodelc");
        println!();
        println!("Or run tests: cargo test --bin coreml_test");
        return;
    }
    
    let model_path = Path::new(&args[1]);
    println!("Testing with model: {}", model_path.display());
    
    // Basic smoke test
    match load_model(model_path) {
        Ok(model) => {
            println!("✅ Model loaded successfully!");
            let model_description = unsafe { model.modelDescription() };
            println!("  Input features: {}", unsafe { model_description.inputDescriptionsByName().count() });
            println!("  Output features: {}", unsafe { model_description.outputDescriptionsByName().count() });
        }
        Err(e) => {
            println!("❌ Model loading failed: {}", e);
        }
    }
    
    // Test tokenizer
    let tokenizer_path = Path::new(TOKENIZER_PATH);
    match load_tokenizer(tokenizer_path) {
        Ok(tokenizer) => {
            println!("✅ Tokenizer loaded successfully!");
            println!("  Vocab size: {}", tokenizer.get_vocab_size(true));
        }
        Err(e) => {
            println!("❌ Tokenizer loading failed: {}", e);
        }
    }
    
    println!();
    println!("Run 'cargo test --bin coreml_test' to run all tests");
    println!("Run 'cargo test --bin coreml_test -- --ignored' to run ignored tests");
}