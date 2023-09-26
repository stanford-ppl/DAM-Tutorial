use dam_macros::{cleanup, identifiable, time_managed};
use dam_rs::{
    channel::{ChannelElement, Receiver, Sender},
    context::Context,
    types::{Cleanable, DAMType},
};
use ndarray::{Array1, Array2};

#[time_managed]
#[identifiable]
pub struct Matmul<T: Clone> {
    weights: Array2<T>,
    biases: Array1<T>,
    input: Receiver<T>,
    output: Sender<T>,
}

impl<T: Clone> Matmul<T> {
    pub fn new(
        input: Receiver<T>,
        output: Sender<T>,
        weights: Array2<T>,
        biases: Array1<T>,
    ) -> Self {
        Self {
            input,
            output,
            weights,
            biases,
            time: Default::default(),
            identifier: Default::default(),
        }
    }
}

impl<T> Context for Matmul<T>
where
    T: DAMType,
{
    fn init(&mut self) {} // Nothing to do here.

    fn run(&mut self) {
        todo!("Implement your Matrix Multiply Block here!")
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

impl<T: Clone> Activation<T> {
    pub fn new(input: Receiver<T>, output: Sender<T>, func: fn(T) -> T) -> Self {
        Self {
            input,
            output,
            func,
            time: Default::default(),
            identifier: Default::default(),
        }
    }
}

impl<T: DAMType> Context for Activation<T> {
    fn init(&mut self) {} // Nothing to do here.

    fn run(&mut self) {
        loop {
            match self.input.dequeue(&mut self.time) {
                dam_rs::channel::Recv::Something(data) => self
                    .output
                    .enqueue(
                        &mut self.time,
                        ChannelElement::new(data.time + 1, (self.func)(data.data)),
                    )
                    .unwrap(),
                dam_rs::channel::Recv::Closed => return,
                _ => unreachable!(),
            }
        }
    }

    #[cleanup(time_managed)]
    fn cleanup(&mut self) {
        self.input.cleanup();
        self.output.cleanup();
    }
}
