import pandas as pd
import model as model
import statsmodels.stats.proportion as smp

# Read the CSV file
df = pd.read_csv('FES_Wiegedaten_seit_2022.csv')

# Filter the columns
filtered_df = df[['GROSS_WEIGHT', 'GROSS_ART']]

# Encode the categorical data
filtered_df['GROSS_ART'] = filtered_df['GROSS_ART'].map({'A': 0, 'H': 1})


def create_classification_accuracy(total_predictions, confidence):

    sample_df = filtered_df.sample(n=total_predictions, replace=True, ignore_index=True)
    sample_df = zip(sample_df['GROSS_WEIGHT'], sample_df['GROSS_ART'])

    total_correct_predictions = 0
    for gross_weight, gross_art in sample_df:
        if model.evaluate(gross_weight) == gross_art:
            total_correct_predictions += 1

    lower, upper = smp.proportion_confint(total_correct_predictions, total_predictions, 1 - confidence)
    print('sample_size=%.3f, lower=%.3f, upper=%.3f' % (total_predictions, lower, upper))


create_classification_accuracy(100_000, 0.95)  # Large sample size
create_classification_accuracy(1000, 0.95)  # Small sample size
