# Sample Data Column Description

This document describes the columns in `sample_data.csv`.

## Columns

### `rootCategory`
The `rootCategory` column represents the highest-level category of the `venueCategory` column from the Foursquare dataset.

#### Example:
If the mapping is given as:

```
Arts and Entertainment > Amusement Park
```

Then the `rootCategory` for "Amusement Park" is "Arts and Entertainment".

### `categoryIndex`
The `categoryIndex` column represents an integer value assigned to each `rootCategory`.

### `surroundingCategories`
The `surroundingCategories` column contains a list of `categoryIndex` values that fall within a randomly determined error range for each row. This error range follows a normal distribution with a mean of 200m and a variance of 50m.