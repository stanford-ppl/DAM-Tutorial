use dam_macros::{cleanup, identifiable, time_managed};
use dam_rs::{
    channel::{ChannelElement, Receiver, Sender},
    context::Context,
    types::{Cleanable, DAMType},
};
use ndarray::prelude::*;

#[time_managed]
#[identifiable]
pub struct GEMV<T: Clone> {
    weights: Array2<T>,
    biases: Array1<T>,
    input: Receiver<T>,
    output: Sender<T>,
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
    ) -> Self {
        let result = Self {
            input,
            output,
            weights,
            biases,
            time: Default::default(),
            identifier: Default::default(),
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
    fn init(&mut self) {} // Nothing to do here.

    fn run(&mut self) {
        let input_size = self.weights.ncols();
        let output_size = self.weights.nrows();
        loop {
            let mut input_vec = Vec::with_capacity(input_size);
            for _ in 0..input_size {
                match self.input.dequeue(&mut self.time) {
                    dam_rs::channel::DequeueResult::Something(data) => {
                        input_vec.push(data.data);
                    }
                    dam_rs::channel::DequeueResult::Closed => return,
                }
                self.time.incr_cycles(1);
            }
            let input_vec = ndarray::Array::from_vec(input_vec);
            let output = self.weights.dot(&input_vec);
            for i in 0..output_size {
                let cur_time = self.time.tick();
                self.output
                    .enqueue(
                        &mut self.time,
                        ChannelElement::new(cur_time + 1 + (i as u64), output[i] + self.biases[i]),
                    )
                    .unwrap();
            }
        }
    }

    #[cleanup(time_managed)]
    fn cleanup(&mut self) {
        self.input.cleanup();
        self.output.cleanup();
    }
}

#[time_managed]
#[identifiable]
pub struct Activation<T: Clone> {
    input: Receiver<T>,
    output: Sender<T>,
    func: fn(T) -> T,
}

impl<T: DAMType> Activation<T> {
    pub fn new(input: Receiver<T>, output: Sender<T>, func: fn(T) -> T) -> Self {
        let result = Self {
            input,
            output,
            func,
            time: Default::default(),
            identifier: Default::default(),
        };

        result.input.attach_receiver(&result);
        result.output.attach_sender(&result);
        result
    }
}

impl<T: DAMType> Context for Activation<T> {
    fn init(&mut self) {} // Nothing to do here.

    fn run(&mut self) {
        loop {
            match self.input.dequeue(&mut self.time) {
                dam_rs::channel::DequeueResult::Something(data) => self
                    .output
                    .enqueue(
                        &mut self.time,
                        ChannelElement::new(data.time + 1, (self.func)(data.data)),
                    )
                    .unwrap(),
                dam_rs::channel::DequeueResult::Closed => return,
            }
        }
    }

    #[cleanup(time_managed)]
    fn cleanup(&mut self) {
        self.input.cleanup();
        self.output.cleanup();
    }
}
