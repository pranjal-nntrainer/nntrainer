// After model initialization and weight loading
// Test forward pass only
std::cout << "Testing forward pass..." << std::endl;

// Create a single input sample
std::vector<float> input_data(INIT_SEQ_LEN, 1.0f);  // dummy input
std::vector<float> label_data(INIT_SEQ_LEN * vocab_size, 0.0f);  // dummy label

// Run forward pass
try {
    auto output = model->forwarding({input_data}, false);
    std::cout << "Forward pass successful. Output size: " << output.size() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Forward pass failed: " << e.what() << std::endl;
}

// Run backward pass
try {
    model->backwarding({label_data}, 0);  // iteration = 0
    std::cout << "Backward pass successful." << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Backward pass failed: " << e.what() << std::endl;
}
