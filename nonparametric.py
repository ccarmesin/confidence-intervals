import pandas as pd
import model as model

from numpy import percentile

# Read the CSV file
df = pd.read_csv('FES_Wiegedaten_seit_2022.csv')

# Filter the columns
filtered_df = df[['GROSS_WEIGHT', 'GROSS_ART']]

# Encode the categorical data
filtered_df['GROSS_ART'] = filtered_df['GROSS_ART'].map({'A': 0, 'H': 1})


def create_nonparametric_confidence_interval(bootstrap_sample_size, sample_size, confidence):
    scores = []
    for _ in range(bootstrap_sample_size):
        sample_df = filtered_df.sample(n=sample_size, replace=True, ignore_index=True)
        sample_df = zip(sample_df['GROSS_WEIGHT'], sample_df['GROSS_ART'])

        total_correct_predictions = 0
        for gross_weight, gross_art in sample_df:
            if model.evaluate(gross_weight) == gross_art:
                total_correct_predictions += 1

        accuracy = total_correct_predictions / sample_size
        scores.append(accuracy)
    lower_p = (1 - confidence) / 2.0
    lower = max(0.0, percentile(scores, lower_p))
    upper_p = confidence + ((1 - confidence) / 2.0)
    upper = min(1.0, percentile(scores, upper_p))
    print('sample_size=%.3f, lower=%.3f, upper=%.3f' % (bootstrap_sample_size * sample_size, lower, upper))


create_nonparametric_confidence_interval(10, 10000, .95)
create_nonparametric_confidence_interval(10, 100, .95)
