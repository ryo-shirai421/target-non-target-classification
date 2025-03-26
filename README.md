## Introduction
This repository provides an implementation of the method described in the paper "Target and Non-target Category Classification from GPS and Check-in Data".

## Requirements
- Python (>= 3.9)
- Required libraries are listed in `pyproject.toml`.

## Preprocessing
The following preprocessing steps are applied beforehand to prepare the data for use:

1. Convert the Foursquare dataset categories to the highest-level category using the category mapping provided by Foursquare (see [Reference](#reference)).
2. Remove categories with 50 or fewer check-ins.
3. Exclude users with 10 or fewer check-ins.
4. For each row, randomly determine an error range using a normal distribution (mean = 200, variance = 50), then collect the categories that fall within this range.

### Sample Data
A sample dataset is available at `./data/csv/sample_data.csv`. For details on each column, refer to `./data/csv/README.md`.

## Usage
1. Place the preprocessed data in the `./data/csv` directory as `data.csv`.
2. Run the following command to execute the process:

   ```sh
   make all
   ```

This command will perform data loading, preprocessing, training, and evaluation.

## Parameters
- `const.unlabel_cat`: This parameter represents the category that is masked as GPS data.
- `const.sample.frac`: The fraction of check-in data that is masked as GPS data.
- `train.loss.type`:
  - `"NUL"`: The proposed method in the paper.
  - `"PNL"` and `"PUL"`: Baseline methods used for comparison in the paper.

## Reference
- **Foursquare dataset:** [Foursquare dataset link](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)  
  *Dingqi Yang, Daqing Zhang, Vincent W Zheng, and Zhiyong Yu. 2014.*  
  *Modeling user activity preference by leveraging user spatial temporal characteristics in LBSNs.*  
  *IEEE Transactions on Systems, Man, and Cybernetics: Systems 45, 1 (2014), 129–142.*
- **Foursquare category mapping:** [Foursquare category mapping link](https://docs.foursquare.com/data-products/docs/categories)
