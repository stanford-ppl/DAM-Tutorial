use dam::{
    simulation::{InitializationOptionsBuilder, ProgramBuilder, RunOptions},
    utility_contexts::*,
};
use dam_tutorial::blocks::*;

// Fill this in!
fn relu(input: f64) -> f64 {
    input.max(0.0)
}

#[test]
fn matmul_relu_test() {
    // For debugging.
    const PRINT_INSTEAD_OF_CHECK: bool = false;
    const NUM_INPUTS: usize = 32;
    const NUM_FEATURES: usize = 1024;
    const NUM_OUTPUTS: usize = 24;

    let activation_func = relu;

    let mut ctx = ProgramBuilder::default();

    let input_vec = {
        // Code to generate input data.
        // We'll be operating at the scalar granularity

        // This isn't idiomatic rust code, but we're prioritizing readability for non rustaceans here.
        let mut data = vec![];
        for input_num in 0..NUM_INPUTS {
            // operating with feature size of 16 elements.
            for feature_id in 0..NUM_FEATURES {
                let feature_value = input_num * NUM_FEATURES + feature_id;
                data.push(feature_value as f64);
            }
        }
        data
    };

    // A copy of the input matrix used for gold testing later.
    let input_mat =
        ndarray::Array::from_shape_vec((NUM_INPUTS, NUM_FEATURES), input_vec.clone()).unwrap();

    let (input_to_mm_send, input_to_mm_recv) = ctx.bounded(1024);
    ctx.add_child(GeneratorContext::new(
        || input_vec.into_iter(),
        input_to_mm_send,
    ));

    let (mm_to_act_send, mm_to_act_recv) = ctx.bounded(1024);

    // Fill a matrix with all 0.5, halving all inputs
    let weights = ndarray::Array2::from_elem((NUM_OUTPUTS, NUM_FEATURES), 0.5);

    // As before, prioritizing readability to non-rustaceans over being idiomatic.
    let mut bias_vec = vec![];
    for output_id in 0..NUM_OUTPUTS {
        bias_vec.push((output_id as f64) - 2.0);
    }
    let biases = ndarray::Array::from_vec(bias_vec);
    let bias_matrix = biases.clone().into_shape((NUM_OUTPUTS, 1)).unwrap();

    ctx.add_child(GEMV::new(
        input_to_mm_recv,
        mm_to_act_send,
        weights.clone(),
        biases,
        1,
    ));

    let (act_to_output_send, act_to_output_recv) = ctx.bounded(1024);

    ctx.add_child(Activation::new(
        mm_to_act_recv,
        act_to_output_send,
        1,
        activation_func,
    ));

    let mut reference_output = weights.dot(&input_mat.t());

    reference_output += &bias_matrix;

    reference_output.mapv_inplace(activation_func);

    if PRINT_INSTEAD_OF_CHECK {
        ctx.add_child(PrinterContext::new(act_to_output_recv));
        println!("Reference: {:?}", reference_output);
    } else {
        ctx.add_child(CheckerContext::new(
            move || reference_output.t().to_owned().into_iter(),
            act_to_output_recv,
        ))
    }

    let executed = ctx
        .initialize(
            InitializationOptionsBuilder::default()
                .run_flavor_inference(true)
                .build()
                .unwrap(),
        )
        .unwrap()
        .run(RunOptions::default());
    println!("Took {:?} cycles", executed.elapsed_cycles());
}
