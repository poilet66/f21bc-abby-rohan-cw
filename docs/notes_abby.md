# Abby's Notes

# 25/09/24

> ## Changes
>
> > - Made some necessary changes to perceptron class
> > - Changed the activation functions into functions rather than classes.
> > - Implemented an ANN (Unsure whether to use the for loop or reduce so done both will ask in lab)
> > - Implemented `MAE Evaluation Metric`
> > - Started looking into testing with dataset

> ## Considerations
>
> > - Might have to change things about as right now it can only handle a single output as a time but will need to handle batches of data to be used with dataset.
> >   > - Instead of 1D array should be a matrix where each row is an input.
> >   > - Should use matrix operations instead of vector.
> >   > - Once we fix this we _should_ be able to get rig of for loop i think?
> > - Unsure if dataset will need preprocessed or if thats been done before it was uploaded?

# 27/09/24

> ## Changes
>
> > - Updated the perceptron class to handle batches of inputs rather than just one input at a time.
> > - The forward pass methods (`forward_for` and `forward_reduce`) now process multiple inputs using matrices.
> > - Edited the activation functions to work with batches of data.
> > - Updated the perceptron’s computations to use matrices instead of vectors for batch processing.
> > - Kept both `for-loop` and `reduce forward` pass methods, will ask lecturer which is best.
> > - The `mean_absolute_error` function now works with batches.
> > - Tested with a small batch of inputs (batch size = 3).(Before I added the saving of test results sorry :()
> > - **Preprocessing the Dataset**:
> >   > - **Standardisation:** Applied Z-score normalisation to ensure each feature has a mean of 0 and standard deviation of 1.
> >   >   > - Z-score was picked because it helps when features follow a normal distribution and makes the model training smoother by putting all features on the same scale.
> >   > - **Outlier Capping:** Used the IQR method to find and cap outliers instead of removing them.
> >   >   > - Capping outliers kept more data without letting extreme values skew the results.
> >   > - **Train-Test Split:** Split the data into 70% training and 30% testing to fairly evaluate model performance.
> > - **Outputting Results:**
> >   > - Added outputs to track test results.
> >   > - The results file now includes:
> >   >   > - Mean Absolute Error (MAE) for both training and test data, calculated for `for loop` and `reduce` methods separately.
> >   >   > - The first 5 predictions (along with actual values) for both training and testing.
> >   >   > - Each test run creates a new file to keep track of results over time.

> ## Considerations
>
> > - Might need to adjust how batch size interacts with the dataset.
> > - The high MAE is expected for now due to random initial weights and no learning yet (backpropagation or optimisation hasn't been implemented).
> >   > - Should see improvements once PSO is added.
