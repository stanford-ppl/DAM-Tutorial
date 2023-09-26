use dam_rs::{
    context::{consumer_context::PrinterContext, generator_context::GeneratorContext},
    simulation::Program,
};
use dam_tutorial::blocks::{Activation, Matmul};

// Fill this in!
fn relu(input: f32) -> f32 {
    return input.max(0.0);
}

#[test]
fn matmul_relu_test() {
    const NUM_INPUTS: usize = 64;
    const NUM_FEATURES: usize = 16;
    const NUM_OUTPUTS: usize = 4;

    let mut ctx = Program::default();

    let (input_to_mm_send, input_to_mm_recv) = ctx.bounded(1024);
    ctx.add_child(GeneratorContext::new(
        || {
            // Code to generate input data.
            // We'll be operating at the scalar granularity

            // This isn't idiomatic rust code, but we're prioritizing readability for non rustaceans here.
            let mut data = vec![];
            for input_num in 0..NUM_INPUTS {
                // operating with feature size of 16 elements.
                for feature_id in 0..NUM_FEATURES {
                    let feature_value = input_num * NUM_FEATURES + feature_id;
                    data.push(feature_value as f32);
                }
            }

            data.into_iter()
        },
        input_to_mm_send,
    ));

    let (mm_to_act_send, mm_to_act_recv) = ctx.bounded(1024);

    // Fill a matrix with all 0.5, halving all inputs
    let weights = ndarray::Array2::from_elem((NUM_FEATURES, NUM_OUTPUTS), 0.5);

    // As before, prioritizing readability to non-rustaceans over being idiomatic.
    let mut bias_vec = vec![];
    for output_id in 0..NUM_OUTPUTS {
        bias_vec.push((output_id as f32) - 2.0);
    }
    let biases = ndarray::Array::from_vec(bias_vec);

    ctx.add_child(Matmul::new(
        input_to_mm_recv,
        mm_to_act_send,
        weights,
        biases,
    ));

    let (act_to_output_send, act_to_output_recv) = ctx.bounded(1024);

    ctx.add_child(Activation::new(mm_to_act_recv, act_to_output_send, relu));

    ctx.add_child(PrinterContext::new(act_to_output_recv));
}
