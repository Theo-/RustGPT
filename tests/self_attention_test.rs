use llm::self_attention::SelfAttention;
use llm::{EMBEDDING_DIM, Layer};
use ndarray::Array2;

#[test]
fn test_self_attention_forward() {
    // Create self-attention module
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // Create input tensor (batch_size=1, seq_len=3, embedding_dim=EMBEDDING_DIM)
    let input = Array2::ones((3, EMBEDDING_DIM));

    // Test forward pass
    let output = self_attention.forward(&input);

    // Check output shape - should be same as input
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_self_attention_backward_respects_scaling() {
    let seq_len = 2;

    let small_dim = 4;
    let mut small_attention = SelfAttention::new_with_weights(
        small_dim,
        Array2::eye(small_dim),
        Array2::eye(small_dim),
        Array2::eye(small_dim),
    );

    let mut small_input = Array2::<f32>::zeros((seq_len, small_dim));
    small_input[[0, 0]] = 1.0;
    small_input[[1, 1]] = 1.0;

    let mut small_grads = Array2::<f32>::zeros((seq_len, small_dim));
    small_grads[[0, 0]] = 1.0;
    small_grads[[1, 1]] = -1.0;

    small_attention.forward(&small_input);
    let small_grad_input = small_attention.backward(&small_grads, 0.0);

    let large_dim = 16;
    let mut large_attention = SelfAttention::new_with_weights(
        large_dim,
        Array2::eye(large_dim),
        Array2::eye(large_dim),
        Array2::eye(large_dim),
    );

    let mut large_input = Array2::<f32>::zeros((seq_len, large_dim));
    large_input[[0, 0]] = 1.0;
    large_input[[1, 1]] = 1.0;

    let mut large_grads = Array2::<f32>::zeros((seq_len, large_dim));
    large_grads[[0, 0]] = 1.0;
    large_grads[[1, 1]] = -1.0;

    large_attention.forward(&large_input);
    let large_grad_input = large_attention.backward(&large_grads, 0.0);

    let small_component = small_grad_input[[1, 0]].abs();
    let large_component = large_grad_input[[1, 0]].abs();

    assert!(
        small_component > large_component * 1.5,
        "Expected gradients for smaller embedding dim to be larger; got small={} and large={}",
        small_component,
        large_component
    );
}

#[test]
fn test_self_attention_with_different_sequence_lengths() {
    // Create self-attention module
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);

    // Test with different sequence lengths
    for seq_len in 1..5 {
        // Create input tensor
        let input = Array2::ones((seq_len, EMBEDDING_DIM));

        // Test forward pass
        let output = self_attention.forward(&input);

        // Check output shape
        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
    }
}
