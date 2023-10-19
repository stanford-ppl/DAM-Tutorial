use dam::context_tools::*;
use ndarray::prelude::*;

/// A simple definition of a GEMV operation
#[context_macro]
pub struct GEMV<T: Clone> {
    weights: Array2<T>,
    biases: Array1<T>,
    input: Receiver<T>,
    output: Sender<T>,
    initiation_interval: u64,
}

impl<T: DAMType> GEMV<T>
where
    T: ndarray::LinalgScalar,
{
    pub fn new(
        input: Receiver<T>,
        output: Sender<T>,
        weights: Array2<T>,
        biases: Array1<T>,
        initiation_interval: u64,
    ) -> Self {
        let result = Self {
            input,
            output,
            weights,
            biases,
            initiation_interval,
            context_info: Default::default(),
        };
        result.input.attach_receiver(&result);
        result.output.attach_sender(&result);
        result
    }
}

impl<T> Context for GEMV<T>
where
    T: DAMType + ndarray::LinalgScalar,
{
    fn run(&mut self) {
        let input_size = self.weights.ncols();
        let output_size = self.weights.nrows();
        loop {
            let mut input_buffer = Vec::with_capacity(input_size);
            // Dequeue repeatedly into the input_vec buffer, at the rate of at most 1 per cycle.
            for i in 0..input_size {
                match self.input.dequeue(&self.time) {
                    Ok(data) => {
                        input_buffer.push(data.data);
                    }
                    // Only legal to break if we don't have anything in the buffer
                    // Otherwise, highly likely to have a book-keeping error.
                    Err(_) if i == 0 => return,
                    Err(_) => panic!("Premature exit encountered!"),
                }
                self.time.incr_cycles(1);
            }

            // After reading the vector, we then calculate all of the values
            let input_vec = ndarray::Array::from_vec(input_buffer);
            let output = self.weights.dot(&input_vec);

            // Output the values at the rate of at most 1/cycle, accounting for backpressure.
            for i in 0..output_size {
                let cur_time = self.time.tick();
                self.output
                    .enqueue(
                        &self.time,
                        ChannelElement::new(cur_time + 1 + (i as u64), output[i] + self.biases[i]),
                    )
                    .unwrap();
            }
            self.time.incr_cycles(self.initiation_interval)
        }
    }
}

#[context_macro]
pub struct Activation<T: Clone> {
    input: Receiver<T>,
    output: Sender<T>,
    initiation_interval: u64,
    func: fn(T) -> T,
}

impl<T: DAMType> Activation<T> {
    pub fn new(
        input: Receiver<T>,
        output: Sender<T>,
        initiation_interval: u64,
        func: fn(T) -> T,
    ) -> Self {
        let result = Self {
            input,
            output,
            initiation_interval,
            func,
            context_info: Default::default(),
        };

        result.input.attach_receiver(&result);
        result.output.attach_sender(&result);
        result
    }
}

/// A Simple Context that just applies a function to its inputs and passes it on.
impl<T: DAMType> Context for Activation<T> {
    fn run(&mut self) {
        loop {
            match self.input.dequeue(&self.time) {
                Ok(data) => self
                    .output
                    .enqueue(
                        &self.time,
                        ChannelElement::new(data.time + 1, (self.func)(data.data)),
                    )
                    .unwrap(),
                Err(_) => return,
            }
            self.time.incr_cycles(self.initiation_interval)
        }
    }
}
