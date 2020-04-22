{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "classifier_models.py",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/chinmayee95/Wifi-localisation/blob/master/classifier_models.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2au2nZxFGgNu",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def classifier_models(traindata,trainlabel):\n",
        "\n",
        "  #Machine Learning Algorithm (MLA) Selection and Initialization\n",
        "  MLA = [\n",
        "      #Ensemble Methods\n",
        "      ensemble.AdaBoostClassifier(),\n",
        "      ensemble.BaggingClassifier(),\n",
        "      ensemble.ExtraTreesClassifier(),\n",
        "      ensemble.GradientBoostingClassifier(),\n",
        "      ensemble.RandomForestClassifier(),\n",
        "\n",
        "      #Gaussian Processes\n",
        "      gaussian_process.GaussianProcessClassifier(),\n",
        "      \n",
        "      #GLM\n",
        "      linear_model.LogisticRegressionCV(),\n",
        "      linear_model.PassiveAggressiveClassifier(),\n",
        "      linear_model.RidgeClassifierCV(),\n",
        "      linear_model.SGDClassifier(),\n",
        "      linear_model.Perceptron(),\n",
        "      \n",
        "      #Navies Bayes\n",
        "      naive_bayes.BernoulliNB(),\n",
        "      naive_bayes.GaussianNB(),\n",
        "      \n",
        "      #Nearest Neighbor\n",
        "      neighbors.KNeighborsClassifier(),\n",
        "      \n",
        "      #SVM\n",
        "      svm.SVC(probability=True),\n",
        "      svm.NuSVC(probability=True),\n",
        "      svm.LinearSVC(),\n",
        "      \n",
        "      #Trees    \n",
        "      tree.DecisionTreeClassifier(),\n",
        "      tree.ExtraTreeClassifier(),\n",
        "      \n",
        "      #Discriminant Analysis\n",
        "      discriminant_analysis.LinearDiscriminantAnalysis(),\n",
        "      discriminant_analysis.QuadraticDiscriminantAnalysis(),\n",
        "\n",
        "      \n",
        "      #xgboost: http://xgboost.readthedocs.io/en/latest/model.html\n",
        "      XGBClassifier()    \n",
        "      ]\n",
        "\n",
        "\n",
        "      #split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit\n",
        "  #note: this is an alternative to train_test_split\n",
        "  cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%\n",
        "\n",
        "  #create table to compare MLA metrics\n",
        "  MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']\n",
        "  MLA_compare = pd.DataFrame(columns = MLA_columns)\n",
        "\n",
        "  #create table to compare MLA predictions\n",
        "  MLA_predict = x_label.copy()\n",
        "\n",
        "  #index through MLA and save performance to table\n",
        "  row_index = 0\n",
        "  for alg in MLA:\n",
        "      #set name and parameters\n",
        "      MLA_name = alg.__class__.__name__\n",
        "      MLA_compare.loc[row_index, 'MLA Name'] = MLA_name\n",
        "      MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())\n",
        "      \n",
        "      #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate\n",
        "      cv_results = model_selection.cross_validate(alg, x_data, x_label, cv  = cv_split,return_train_score=True)\n",
        "\n",
        "      MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()\n",
        "      MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()\n",
        "      MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   \n",
        "      #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets\n",
        "      MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!\n",
        "      \n",
        "\n",
        "      #save MLA predictions - see section 6 for usage\n",
        "      alg.fit(x_data, x_label)\n",
        "      MLA_predict[MLA_name] = alg.predict(x_data)\n",
        "      \n",
        "      row_index+=1\n",
        "\n",
        "      \n",
        "  #print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html\n",
        "  MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)\n",
        "  MLA_compare\n",
        "  #MLA_predict\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "P299AmaWG2le",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}