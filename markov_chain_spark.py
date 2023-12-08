from pymongo import MongoClient
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.metrics import r2_score
from pyspark.sql.functions import month, year
import numpy as np
import matplotlib.pyplot as plt
import distinctipy
from math import ceil, sqrt
import pickle
import json
import time
from sklearn.metrics import mean_squared_error


def main():
    # CONNECT TO DB, QUERY FOR DATA COLLECTION, CONVERT TO NUMPY
    array = get_array_from_db()

    # Print the type of 'array_without_outliers' and the top 5 samples for checking
    print("Data as numpy array")
    print(f"Type = {type(array)}")
    print(array[:5])
    print()

    # REMOVE OUTLIERS
    # List of values for m
    ms = [1, 2, 3, 4, 5, 6]
    total_number_samples = plot_results_for_values_m('out/m_values', ms, array)

    # Preferred value of m based on plot
    preferred_m_value = 2
    array_no_outliers = reject_outliers(array[:, 0], preferred_m_value)

    new_sample_count = len(array_no_outliers)
    count_diff = total_number_samples - new_sample_count
    print(f"Original sample count: {total_number_samples}")
    print(f"Downsized to chosen  : {new_sample_count} ({count_diff} removed)")
    plot_hist_of_reduced('out/outliers_removed', total_number_samples, array_no_outliers)

    # GET CHANGE THRESHOLDS FOR STATE TRANSITIONS OF MODEL
    change_array = get_change_array(array_no_outliers)
    print(f"Load changes: shape = {change_array.shape}")
    print(change_array)
    negative_changes, no_changes, positive_changes = get_changes_by_direction(change_array)
    neg_min_value, neg_q1, neg_q2, neg_q3, neg_max_value, \
        pos_min_value, pos_q1, pos_q2, pos_q3, pos_max_value = \
        get_changes_five_summary_stats(negative_changes, positive_changes)
    plot_change_summaries('out/changes', total_number_samples,
                          negative_changes, no_changes, positive_changes,
                          neg_min_value, neg_q1, neg_q2, neg_q3, neg_max_value,
                          pos_min_value, pos_q1, pos_q2, pos_q3, pos_max_value)
    print(f"Negatives: Min = {neg_min_value:.5f}, Q1 = {neg_q1:.5f}, Q2 = {neg_q2:.5f}, "
          f"Q3 = {neg_q3:.5f}, Max = {neg_max_value:.5f}")
    print(f"No changes: count = {len(no_changes)}")
    print(f"Positives: Min = {pos_min_value:.5f}, Q1 = {pos_q1:.5f}, Q2 = {pos_q2:.5f}, "
          f"Q3 = {pos_q3:.5f}, Max = {pos_max_value:.5f}")

    # Number of change states (described above)
    num_changes, num_months = 7, 12
    targets = [neg_q1, neg_q2, neg_q3, 0, pos_q1, pos_q2, pos_q3]
    print(f"All targets:\n{[round(target, 3) for target in targets]}")

    # OUTPUT FILE PATHS
    train_out_path = 'out/train_preds_v_acts'
    train_saved_model_path = 'out/post_train_markov_chain_model'
    test_out_path = 'out/test_preds_v_acts'
    test_saved_model_path = 'out/post_test_markov_chain_model'

    # CREATE AND TRAIN THE MARKOV CHAIN
    # Create an instance of Markov chain for the number of months and number of changes
    markov_chain = MarkovChain(num_changes, num_months, targets)

    # Start processing the data (array) and make a prediction for the next sample's load
    print(f"Starting training")
    start_time = time.time()
    train_preds, train_acts = markov_chain.predict_and_train(train_out_path, array, prnt_every=0)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Finished in {execution_time:.3f} seconds.")
    plot_preds_versus_acts(train_out_path, array, train_preds, train_acts)
    markov_chain.save_instance(train_saved_model_path)

    # Print training results
    train_preds, train_acts = load_results(train_out_path + ".txt")
    print_mse_and_results(train_preds, train_acts, 1000)

    # TESTING
    test_df = get_testing_df("./data/test_dataframes.xlsx")
    test_array = test_df.values
    print(f"{type(test_array)}: shape = {test_array.shape}")
    print(test_array[:5])
    print(...)
    print(test_array[-5:])

    markov_chain = load_instance(train_saved_model_path)
    print(f"Starting testing")
    start_time = time.time()
    test_preds, test_acts = markov_chain.predict_and_train(test_out_path, test_array, prnt_every=0)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Finished in {execution_time:.3f} seconds.")

    if test_preds is None:
        test_preds, test_acts = load_results(test_out_path + ".txt")
    print_mse_and_results(test_preds, test_acts, 10)
    markov_chain.save_instance(test_saved_model_path)

def print_mse_and_results(preds, acts, print_every):
    print("Sample index: <predicted load> vs. <actual load> (<percentage diff> %)")
    if print_every > 0:
        for i, pred in enumerate(preds):
            if i % print_every == 0:
                print(f"{i}: {pred:.2f} vs. {acts[i]:.2f} ({abs(pred - acts[i]) / acts[i] * 100:.2f} % diff)")
    mse_markov = mean_squared_error(acts, preds)
    rmse_markov = sqrt(mse_markov)
    print()
    print(f"MSE = {mse_markov}")
    print(f"RMSE = {rmse_markov}")


def get_array_from_db():
    print(f"Connecting to and retrieving data from the database.")
    start_time = time.time()
    client, db, collection = init_db_connection()
    mongo_data = retrieve_data(collection)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"DB retrieval finished in {execution_time:.3f} seconds.")

    print(f"Pre-processing data.")
    start_time = time.time()
    pandas_df = convert_data_to_dataframe(mongo_data)
    drop_dataframe_column('_id', pandas_df)
    spark = init_spark_session()
    spark_df = convert_pandas_df_to_spark(spark, pandas_df)
    df_month_year = modify_spark_dataframe(spark_df)
    array = convert_spark_df_to_numpy(df_month_year)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Data pre-processing finished in {execution_time:.3f} seconds.")
    spark.stop()
    return array


def get_testing_df(file_path):
    # Read the new data
    df = pd.read_excel(file_path)
    # Keep only the columns 'datetime' and 'nat_demand'
    df = df[['datetime', 'DEMAND']]

    # Make a copy of the dataframe
    df_month_year = df.copy()

    # Change column 'datetime' to type datetime
    df_month_year['datetime'] = pd.to_datetime(df_month_year['datetime'])

    # Replace datetime values with month numbers
    df_month_year['month'] = df_month_year['datetime'].dt.month

    # Add new column with the year
    df_month_year['year'] = df_month_year['datetime'].dt.year

    # Remove datetime column since no longer needed
    return df_month_year.drop('datetime', axis=1)


def load_results(save_path):
    with open(save_path, 'r') as f:
        json_str = f.read()
    return json.loads(json_str)


def init_db_connection():
    # Initialize MongoDB Connection
    client = MongoClient('mongodb+srv://JTXBigData:pJRAyKW9QnqE7B1G@jtxbigdatacluster.dzo50pn.mongodb.net/')
    db = client['JTXBigDataCluster']
    # collection = db['jtx-reduced-data']
    collection = db['training-flattened']
    return client, db, collection


def retrieve_data(collection):
    return list(collection.find())


def convert_data_to_dataframe(mongo_data):
    return pd.DataFrame(mongo_data)


def drop_dataframe_column(name, pandas_df):
    # Drop the _id column provided by MongoDB
    if name in pandas_df.columns:
        pandas_df.drop(name, axis=1, inplace=True)


def init_spark_session():
    SparkSession.builder.config("spark.driver.host", "localhost").getOrCreate()
    return SparkSession.builder.appName("MongoDBToSparkDF").getOrCreate()


def convert_pandas_df_to_spark(spark, pandas_df):
    return spark.createDataFrame(pandas_df)


def modify_spark_dataframe(spark_df):
    # Assuming spark_df is a Spark DataFrame
    # Keep only the columns 'datetime' and 'DEMAND'
    df = spark_df.select('datetime', 'DEMAND')
    
    # Change column 'datetime' to type datetime
    df = df.withColumn('datetime', df['datetime'].cast('timestamp'))

    # NEW LINE FOR TESTING ERROR
    df_sorted = df.sort(['datetime'])

    # Add new columns with month and year
    df_month_year = df_sorted.withColumn('month', month('datetime')).withColumn('year', year('datetime'))

    # Drop the 'datetime' column since it's no longer needed
    df_month_year = df_month_year.drop('datetime')
   
    return df_month_year


# Function found online to remove outliers based on choice of m
def reject_outliers(data, m):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def convert_spark_df_to_numpy(df_month_year):
    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df_month_year = df_month_year.toPandas()

    # Convert Pandas DataFrame to NumPy array
    array = pandas_df_month_year.values
    return array


def plot_results_for_values_m(save_path, ms, array):
    # Get total number of samples
    total_number_samples = array.shape[0]

    # Array of sample values for each different subset of the entire dataset
    arrays_no_outliers = []

    # Loop through different values of m
    for m in ms:
        # Use the reject_outliers function to remove outliers for the specified column ('DEMAND' in this case)
        array_subset = reject_outliers(array[:, 0], m)
        arrays_no_outliers.append(array_subset)
        print(f"m = {m}: {len(array_subset)} samples ({total_number_samples - len(array_subset)} \"outliers\" removed)")
    print()

    # Boxplot with each sample subset distribution represented by a different box
    fig, ax = plt.subplots()

    # Apply labels on the x-axis for m values
    box = ax.boxplot(arrays_no_outliers, patch_artist=True, labels=ms)

    # Get unique color for each m value (in ms)
    colors = distinctipy.get_colors(len(ms))

    # Apply the colors to the boxplot boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Label the plot
    plt.title("Box plots of sample selections by values of m")
    plt.xlabel("M values")
    plt.ylabel("Load values")
    plt.savefig(save_path + ".png", format='png')
    plt.show()
    plt.clf()
    return total_number_samples


def plot_hist_of_reduced(save_path, total_number_samples, array_no_outliers):
    num_bins = ceil(total_number_samples / 10)
    plt.hist(array_no_outliers, bins=num_bins)
    plt.title("Histogram of new sampling's load values")
    plt.xlabel("Load values")
    plt.ylabel("Number of samples")
    plt.savefig(save_path + ".png", format='png')
    plt.show()
    plt.clf()


def get_change_array(array_no_outliers):
    # Get the demand (load) as list for easy list comprehension
    column_as_list = array_no_outliers.tolist()
    total_number_samples = len(column_as_list)
    # Create a new list of load changes between the next index and previous via list comprehension
    change_list = [column_as_list[i + 1] - column_as_list[i] for i in range(total_number_samples - 1)]
    return np.array(change_list)


def get_changes_by_direction(change_array):
    negative_changes = change_array[change_array < 0]
    no_changes = change_array[change_array == 0]
    positive_changes = change_array[change_array > 0]
    return negative_changes, no_changes, positive_changes


def get_changes_five_summary_stats(negative_changes, positive_changes):
    # Get the min and max values for setting the plot x range
    neg_min_value = np.min(negative_changes)
    neg_max_value = np.max(negative_changes)

    pos_min_value = np.min(positive_changes)
    pos_max_value = np.max(positive_changes)

    # Calculate Q1, Q2, Q3
    neg_q1 = np.percentile(negative_changes, 25)
    neg_q2 = np.percentile(negative_changes, 50)
    neg_q3 = np.percentile(negative_changes, 75)

    pos_q1 = np.percentile(positive_changes, 25)
    pos_q2 = np.percentile(positive_changes, 50)
    pos_q3 = np.percentile(positive_changes, 75)

    return neg_min_value, neg_q1, neg_q2, neg_q3, neg_max_value, \
        pos_min_value, pos_q1, pos_q2, pos_q3, pos_max_value


def plot_change_summaries(save_path, total_number_samples, negative_changes, no_changes, positive_changes,
                          neg_min_value, neg_q1, neg_q2, neg_q3, neg_max_value,
                          pos_min_value, pos_q1, pos_q2, pos_q3, pos_max_value):

    # Set the number of bins for the histogram
    num_bins = ceil(total_number_samples / 10)
    if num_bins > 500:
        num_bins = 500
    
    # Create a histogram of the array
    n_neg, bins_neg, patches_neg = plt.hist(negative_changes, bins=num_bins, color='darkred', alpha=0.5, label='Negative Changes')
    n_no, bins_no, patches_no = plt.hist(no_changes, bins=num_bins, color='gray', alpha=0.5, label='No Changes')
    n_pos, bins_pos, patches_pos = plt.hist(positive_changes, bins=num_bins, color='darkgreen', alpha=0.5, label='Positive Changes')
    
    max_bin_height = max(max(n_neg), max(n_no), max(n_pos))
    # plt.ylim(0, max_bin_height * 1.1)
    plt.ylim(0, 500)  # hard coding b/c plotting y range way too big for some reason
    
    # Create vertical lines at each of Five-number summary
    neg_line_color = 'orange'
    plt.axvline(x=neg_min_value, color=neg_line_color)
    plt.axvline(x=neg_q1, color=neg_line_color)
    plt.axvline(x=neg_q2, color=neg_line_color)
    plt.axvline(x=neg_q3, color=neg_line_color)
    plt.axvline(x=neg_max_value, color=neg_line_color)

    zero_line_color = 'gray'
    plt.axvline(x=0, color=zero_line_color)

    pos_line_color = 'turquoise'
    plt.axvline(x=pos_min_value, color=pos_line_color)
    plt.axvline(x=pos_q1, color=pos_line_color)
    plt.axvline(x=pos_q2, color=pos_line_color)
    plt.axvline(x=pos_q3, color=pos_line_color)
    plt.axvline(x=pos_max_value, color=pos_line_color)
    
    # Label plot
    plt.title("Histogram of all load changes")
    plt.xlabel("Load change values")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.savefig(save_path + ".png", format='png')
    plt.show()
    plt.clf()


class State:
    times_in_state = 0  # Number of times the model has been in this state
    total_load = 0  # The running load of all time state is visited
    prev_load = 0  # The last seen load value

    # Class constructor
    def __init__(self, i, j, num_changes, targets):
        self.month = i
        self.change = j
        self.same_month_trans_counts = np.zeros(num_changes)  # transition probabilities for same month changes
        self.next_month_trans_counts = np.zeros(num_changes)  # transition probabilities for next month changes
        self.targets = targets

    # Currently went to this node based on training observation
    def visit(self, load):
        self.times_in_state += 1
        self.total_load += load
        self.prev_load = load

    # Used for predicting load from the previous state to this one
    def get_load(self, cur_load):
        return cur_load + self.targets[self.change]

    # Function to print out Node object instance info for checking
    def print_out(self):
        if self.times_in_state > 0:
            avg = self.total_load / self.times_in_state
        else:
            avg = self.prev_load
        print(f"State: ({self.month}, {self.change}), Visits: {self.times_in_state}, "
              f"Prev load (avg): {self.prev_load} ({avg})\n"
              f"\tMonth {self.month} transitions: {self.same_month_trans_counts}\n"
              f"\tMonth {(self.month + 1) % 12} transitions: {self.next_month_trans_counts}"
              )

    # Returns the top transition change pick based on the current stochastic model
    def get_state_transition_pred(self, same_month):

        # Month is staying the same
        if same_month:
            trans_counts = self.same_month_trans_counts

        # Month is moving to the next month
        else:
            trans_counts = self.next_month_trans_counts

        # Get the index of the top pick (largest number of times moved to that change from this one)
        top_change_pick = np.argmax(trans_counts)

        # If been in this state at least once
        if self.times_in_state > 0:

            # Will have confidence above indifferent (1/num_changes%)
            prob_top_pick = np.max(trans_counts) / self.times_in_state

        else:
            # Confidence is indifferent (same for all possible changes)
            prob_top_pick = 1 / len(self.same_month_trans_counts)

        # Return the top change pick and associated confidence
        return top_change_pick, prob_top_pick


# Class for the Markov Chain model
class MarkovChain:

    # Constructor initializes the needed transition matrix automatically
    def __init__(self, num_changes, num_months, targets):
        self.num_changes = num_changes
        self.num_months = num_months
        self.targets = targets
        self._init_transition_matrix()

    # Initializes a transition matrix based on constructor parameters
    def _init_transition_matrix(self):
        # Create empty array of dimensions number_months by num_changes (12 by 7) to store State object references
        transition_matrix = np.empty((self.num_months, self.num_changes), dtype=State)
        # for each month [0-11]
        for i in range(self.num_months):
            # For each change [0-6]
            for j in range(self.num_changes):
                transition_matrix[i][j] = State(int(i), j, self.num_changes, self.targets)
        # Initialize instance variable to created matrix
        self.transition_matrix = transition_matrix

    # Method to iterate over data samples and make predictions while updating the model to keep track of prior
    # observations/state transitions
    def predict_and_train(self, save_path, sample_set, prnt_every=10000):

        # Get the starting month
        starting_month = int(sample_set[0, 1])

        # Set starting change to NO CHANGE (since no previous information)
        starting_change = 3

        # Get starting state from transition matrix using key scheme
        starting_state = self.transition_matrix[starting_month, starting_change]

        # Get starting load from first sample
        starting_load = sample_set[0, 0]

        # Visit the starting state passing cur load
        starting_state.visit(starting_load)

        # Set prev state as the starting
        prev_state = starting_state

        # set up lists to add predictions to
        preds, trues = [], []

        # For each sample in training set except the first (used to seed start)
        for i, sample in enumerate(sample_set[1:]):  # start at second training sample

            # Offset i to reflect skipping 1
            i += 1

            # Get the cur and prev month and load values
            cur_month = int(sample[1])
            cur_load = sample[0]
            prev_month = prev_state.month
            prev_load = prev_state.prev_load

            # Compute load change
            load_change = cur_load - prev_load

            # Determine which change the current change is closest to target wise
            closest_target = min(self.targets, key=lambda x: abs(x - load_change))

            # Set change transition target to index of the closest value
            change_target = self.targets.index(closest_target)

            # Boolean if month is same between prev and cur
            same_month = False

            # Ensure cur_month is within the valid range [0, 11]
            cur_month = cur_month % self.num_months

            # CHeck if staying in the same month
            if cur_month == prev_month:

                same_month = True
                # Increment change target tally of the state
                prev_state.same_month_trans_counts[change_target] += 1

            # later month
            elif cur_month != prev_month:

                # Check if valid month transition [0-11] and just 1 ahead
                if ((cur_month == (prev_month + 1) % self.num_months) or
                        (cur_month == 0 and prev_month == self.num_months - 1)):

                    # Increment change target tally of the state
                    prev_state.next_month_trans_counts[change_target] += 1
                else:
                    # Check if transitioning from month 4 to month 1
                    if prev_month == 4 and cur_month == 1:
                        # Increment change target tally of the state
                        prev_state.next_month_trans_counts[change_target] += 1
                    else:
                        # Stop execution because training will be invalid
                        error_msg = f"Encountered an invalid forward state transition from {prev_month} -> {cur_month}."
                        print(error_msg)
                        print(f"cur_month: {cur_month}, prev_month: {prev_month}")
                        raise ValueError(error_msg)

            # Get predicted next change and associated probability (confidence)
            predicted_next_change, prob = prev_state.get_state_transition_pred(same_month)

            # Get the target month (month of next sample to predict), same or next
            next_index = (i + 1) % len(sample_set)
            next_sample_month = int(sample_set[next_index][1])

            # Get the predicted next state using predicted change and known target month
            # predicted_next_state = self.transition_matrix[next_sample_month, predicted_next_change]
            predicted_next_state = self.transition_matrix[next_sample_month - 1, predicted_next_change]

            # Get prediction for next load value
            predicted_next_load = predicted_next_state.get_load(cur_load)

            # Add prediction to the running predictions list
            preds.append(predicted_next_load)

            # Get the actual next load
            actual_next_load = sample_set[next_index][0]

            # Add the actual next load to actuals running list
            trues.append(actual_next_load)

            # Compute the percentage difference prediction is from the "true" value
            predict_diff = abs((actual_next_load - predicted_next_load) / actual_next_load * 100)

            # Calculate R-squared
            if len(preds) > 1:
                r2 = r2_score(trues, preds)

            # Metric is considered undefined for single pair
            else:
                r2 = "undefined"

            # Get the state that reflects current load change
            cur_state = self.transition_matrix[cur_month, change_target]
            cur_state.visit(cur_load)

            # Print out extra info by specified iteration cycle for checking
            if prnt_every > 0 and i % prnt_every == 0:
                print(f"i = {i}/{len(sample_set) - 1}")
                print(f"Month: ({prev_month} -> {cur_month}), Load: ({prev_load} -> {cur_load})")
                print(f"Load change: {load_change} -> {closest_target}, Change type: {change_target}")
                print("Prev state:")
                print(f"   Same month transitions: {prev_state.same_month_trans_counts}")
                print(f"   Next month transitions: {prev_state.next_month_trans_counts}")

                print("Cur state:")
                print(f"   Same month transitions: {cur_state.same_month_trans_counts}")
                print(f"   Next month transitions: {cur_state.next_month_trans_counts}")
                print(f"Predicted change: {predicted_next_change}, Actual change: {change_target}")
                print(
                    f"Prediction: {predicted_next_load:.2f}, Actual: {actual_next_load:.2f}, "
                    f"% Difference: {predict_diff:.2f} %")
                print("-----------------------------------------------------------------------------\n")

                # Print out current iteration, relative progress percentage and running r-square measure across the
                # entire set
            print(f"i = {i}/{len(sample_set) - 1} ({i / len(sample_set) * 100:.2f} %): r2 = {r2}", end="\r")

            # Set previous state to current state for next iteration
            prev_state = cur_state

        results_tuple = (preds, trues)
        # Convert the tuple to a JSON string
        json_str = json.dumps(results_tuple)
        # Save the JSON string to a file
        with open(save_path + ".txt", 'w') as f:
            f.write(json_str)

        # once all samples have been processed, return predictions and associated true load values
        return preds, trues

    def save_instance(self, save_path):
        # Save the instance to a file
        file_path = save_path + '.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Markov Chain instance saved to '{file_path}'.")


def load_instance(load_path):
    file_path = load_path + '.pkl'
    with open(file_path, 'rb') as f:
        instance = pickle.load(f)
    print(f"Instance loaded from '{file_path}'.")
    return instance


# Function to plot the preds vs actual load values
def plot_preds_versus_acts(save_path, array, preds, acts):
    # Get unique years in the dataset and count per year
    years, year_counts = np.unique(array[:, 2], return_counts=True)

    print(f"Years: {years}")
    print(f"Counts: {year_counts}")

    # generate visually distinct colors for each year
    colors = distinctipy.get_colors(len(years))
    line_width = 0.5
    plt.plot(preds, label='Predictions', linewidth=line_width)
    plt.plot(acts, label='Actual', linewidth=line_width)
    plt.legend()
    plt.title("Predictions vs. Actual load values")
    plt.xlabel("Sample Index")
    plt.ylabel("Load value")

    # Place vertical lines between each sample group of same year
    index = 0
    for i, y in enumerate(years):
        index += year_counts[i]
        plt.axvline(x=index, color=colors[i], label=y)
    plt.savefig(save_path + ".png", format='png')
    # plt.show()
    plt.clf()

if __name__ == "__main__":
    main()
