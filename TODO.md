# TODO

## Essential

- [x] Implement other activation functions (ReLu & Tanh)
- [x] Implement ANN (layers of perceptrons that feed into each other)
- [x] Implement an evaluation metric
- [x] Fix code so it works with batches instead of just a single input
- [x] Preprocess dataset
- [x] Get dataset working with existing ANN
- [x] Test ANN on a regression task
- [ ] Implement PSO Algorithm
    - [ ] Create a way of either updating weights/biases in existing ANN OR
    - [ ] Way to instantiate NEW ANN with predetermined weights/biased
    - [ ] Specify N-dimensional space of all possible weights/biases (each axis' range = range of weight/bias) (explore what these are)
    - [ ] Create particle class
        - [ ] Direction
        - [ ] Direction change algorithm
    - [ ] Link particle to ANN/forward pass algorithm to determine loss of particle position
    - [ ] Update/iteration system
    - [ ] Return optimal space
- [ ] Test PSO on a regression task (maybe a simpler task instead/as well as?)
- [ ] Combine PSO & ANN
- [ ] Test combined on a regression task

## Might be nice

- [x] Lambda implementation of perceptron (more extensibility)

## Quality of Life

- [ ] Virtual environment for package/vs code extensions?
- [ ] Split classes into separate files
