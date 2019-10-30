# Radix.ai Machine Learning Challenge

## Introduction

The goal of this challenge is to build a Machine Learning model to predict the genres of a movie, given its synopsis. Your solution will be evaluated on the performance of your Machine Learning model and on the quality of your code.

To succeed, you must implement a Python package called `challenge`, which exposes a Flask REST API according to the REST API specification in `api.yml` (you can view this specification at [editor.swagger.io](https://editor.swagger.io)).

Specifically, your Flask REST API must expose:
1. A training endpoint at `localhost:5000/genres/train` to which you can POST a CSV with header `movie_id,synopsis,genres`, where `genres` is a space-separated list of movie genres.
2. A prediction endpoint at `localhost:5000/genres/predict` to which you can POST a CSV with header `movie_id,synopsis` and returns a CSV with header `movie_id,predicted_genres`, where `predicted_genres` is a space-separated list of the top 5 movie genres.

Please note that unfortunately we cannot offer visa sponsorship.

## Getting started

### Kaggle setup

1. Go to the [Kaggle competition for this challenge](https://www.kaggle.com/t/13e0d7502d7746459ef8e4c594a6219a) and click on `Join Competition`.
2. Go to your Kaggle `My Account` page and click on `Create New API Token` to download your Kaggle authentication file `kaggle.json`.

### GitLab setup

1. [Fork this repo](https://gitlab.com/radix-ai/challenge/forks/new).
2. In your forked challenge repo:
   1. Go to `Settings > General > Visibility [...]` and set the project visibility to `Private`.
   2. Go to `Settings > Members > Invite Member` and add `radix-ai-developers` as a `Reporter` so we can follow along with your progress.
   3. Go to `Settings > CI / CD > Variables` and:
      1. Add a `KAGGLE_USERNAME` key with your Kaggle username as value.
      2. Add a `KAGGLE_KEY` key with the key from the `kaggle.json` you downloaded in the previous section.

### Local setup

1. Install [Miniconda](https://conda.io/miniconda.html) if you don't have it already.
2. Run `conda env create` from the repo's base directory to create the repo's conda environment from `environment.yml`.
3. Run `conda activate challenge-env` to activate the conda environment.

## Running and evaluating your solution

Every time you push your code to GitLab your solution will automatically be run and evaluated in a GitLab Pipeline:

1. First, `train_predict.sh` will:
    1. Create and activate the Conda environment.
    2. Download the dataset from Kaggle.
    3. Start your Flask API as `env FLASK_APP=challenge flask run`
    4. POST `train.csv` to `localhost:5000/genres/train` to train your model.
    5. POST `test.csv` to `localhost:5000/genres/predict` to create a `submission.csv` with the top 5 predicted genres for each test movie.
    6. Upload `submission.csv` to the [Kaggle competition](https://www.kaggle.com/t/13e0d7502d7746459ef8e4c594a6219a) (on GitLab only).
2. Then, `score.sh` will:
    1. Get your submission's Kaggle score, which is the [Mean Average Precision at K](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29) of the top 5 predicted genres.
    2. Compute a score that indicates the quality of your code.
    3. Print your final score, which is the geometric mean of (1) and (2).

You can run these scripts locally as much as you want using `./train_predict.sh` and `./score.sh`. We will evaluate your final submission by inspecting your last successful GitLab Pipeline.

## Good luck!

-- radix.ai
