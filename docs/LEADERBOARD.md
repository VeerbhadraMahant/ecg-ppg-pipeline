# Final leak-free leaderboard

All thresholds fixed at 0.5; every model sees identical splits and the seeds shared by all models in a table. Hyper-parameters / combiners were fitted on inner validation sets only. CI = paired record-level bootstrap of (best - model).

## Challenge 2015 (PPG cohort, 5-fold CV)
*seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; best by accuracy: **Ensemble (logit mean, 10 models)***

| Model | Group | Accuracy | F1 | AUC | Sens | Spec | Challenge | best−model ΔF1 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Ensemble (logit mean, 10 models) | ensemble | 0.876 | 0.827 | 0.948 | 0.795 | 0.926 | 67.5 | - |
| XGBoost on features v2 | classical | 0.871 | 0.818 | 0.939 | 0.782 | 0.925 | 66.2 | +0.008 [-0.007, +0.022] |
| Extra Trees on features v2 | classical | 0.867 | 0.802 | 0.947 | 0.725 | 0.953 | 61.8 | +0.024 [+0.001, +0.047] |
| Ensemble (LR stacker) | ensemble | 0.864 | 0.825 | 0.945 | 0.856 | 0.870 | 71.5 | +0.002 [-0.013, +0.019] |
| Random Forest on features v2 | classical | 0.863 | 0.795 | 0.938 | 0.717 | 0.952 | 61.0 | +0.031 [+0.010, +0.052] |
| MLP on features v2 | classical | 0.857 | 0.802 | 0.927 | 0.778 | 0.906 | 64.8 | +0.025 [+0.008, +0.042] |
| SVM (RBF) on features v2 | classical | 0.852 | 0.791 | 0.928 | 0.762 | 0.909 | 63.6 | +0.034 [+0.020, +0.048] |
| Logistic regression on features v2 | classical | 0.852 | 0.801 | 0.928 | 0.799 | 0.886 | 65.9 | +0.026 [+0.010, +0.044] |
| HistGB on features v1 | classical | 0.851 | 0.798 | 0.918 | 0.788 | 0.891 | 65.0 | +0.029 [+0.004, +0.054] |
| ResNet1D (deep) | deep | 0.820 | 0.760 | 0.884 | 0.764 | 0.854 | 61.4 | +0.067 [+0.041, +0.093] |
| TCN (deep) | deep | 0.810 | 0.751 | 0.887 | 0.765 | 0.838 | 60.5 | +0.076 [+0.051, +0.102] |
| InceptionTime (deep) | deep | 0.796 | 0.732 | 0.876 | 0.745 | 0.827 | 58.3 | +0.095 [+0.068, +0.122] |
| CNN cross-attention fusion (proposed) | deep | 0.765 | 0.699 | 0.841 | 0.734 | 0.786 | 55.4 | +0.128 [+0.094, +0.161] |
| CNN concat fusion (deep) | deep | 0.757 | 0.693 | 0.836 | 0.731 | 0.773 | 54.4 | +0.134 [+0.098, +0.171] |

## Challenge 2015 (PPG cohort, 5 seeds, incl. hybrid deep recipe)
*seeds: [0, 1, 2, 3, 4]; best by accuracy: **Ensemble incl. hybrid deep (mean)***

| Model | Group | Accuracy | F1 | AUC | Sens | Spec | Challenge | best−model ΔF1 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Ensemble incl. hybrid deep (mean) | ensemble | 0.878 | 0.830 | 0.950 | 0.795 | 0.928 | 67.6 | - |
| Hybrid deep: attn CNN-RNN + features, aug/EMA/TTA/ens | deep | 0.875 | 0.832 | 0.932 | 0.825 | 0.906 | 69.7 | -0.002 [-0.029, +0.026] |
| Ensemble incl. hybrid deep (LR stacker) | ensemble | 0.875 | 0.839 | 0.948 | 0.870 | 0.878 | 73.5 | -0.008 [-0.026, +0.012] |
| XGBoost on features v2 | classical | 0.870 | 0.816 | 0.938 | 0.775 | 0.926 | 65.6 | +0.013 [-0.005, +0.032] |
| Extra Trees on features v2 | classical | 0.868 | 0.803 | 0.946 | 0.723 | 0.955 | 61.7 | +0.026 [+0.001, +0.049] |
| Random Forest on features v2 | classical | 0.863 | 0.794 | 0.937 | 0.709 | 0.955 | 60.5 | +0.036 [+0.012, +0.058] |
| SVM (RBF) on features v2 | classical | 0.855 | 0.800 | 0.930 | 0.772 | 0.906 | 64.0 | +0.030 [+0.014, +0.047] |
| Logistic regression on features v2 | classical | 0.854 | 0.804 | 0.928 | 0.803 | 0.886 | 66.4 | +0.026 [+0.007, +0.047] |
| MLP on features v2 | classical | 0.853 | 0.797 | 0.925 | 0.771 | 0.903 | 63.9 | +0.033 [+0.013, +0.054] |
| ResNet1D (deep) | deep | 0.830 | 0.778 | 0.891 | 0.790 | 0.855 | 63.6 | +0.053 [+0.025, +0.079] |
| CNN cross-attention fusion (proposed) | deep | 0.774 | 0.713 | 0.842 | 0.754 | 0.787 | 57.1 | +0.117 [+0.084, +0.154] |

## VTaC (official split, 5 seeds)
*seeds: [0, 1, 2, 3, 4]; best by accuracy: **Hybrid deep: attn CNN-RNN + features, aug/EMA/TTA/ens***

| Model | Group | Accuracy | F1 | AUC | Sens | Spec | Challenge | best−model ΔF1 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| Hybrid deep: attn CNN-RNN + features, aug/EMA/TTA/ens | deep | 0.913 | 0.858 | 0.966 | 0.933 | 0.905 | 84.9 | - |
| Ensemble incl. hybrid deep (logit mean) | ensemble | 0.912 | 0.844 | 0.957 | 0.842 | 0.940 | 77.5 | +0.014 [-0.028, +0.058] |
| Ensemble (logit mean, 10 models) | ensemble | 0.910 | 0.839 | 0.955 | 0.833 | 0.940 | 76.7 | +0.019 [-0.023, +0.061] |
| Ensemble incl. hybrid deep (LR stacker) | ensemble | 0.905 | 0.849 | 0.964 | 0.945 | 0.890 | 85.3 | +0.009 [-0.014, +0.033] |
| MLP on features v2 | classical | 0.900 | 0.821 | 0.949 | 0.818 | 0.932 | 74.8 | +0.037 [-0.006, +0.080] |
| XGBoost on features v2 | classical | 0.895 | 0.814 | 0.942 | 0.823 | 0.922 | 74.7 | +0.043 [+0.002, +0.086] |
| SVM (RBF) on features v2 | classical | 0.891 | 0.806 | 0.948 | 0.803 | 0.926 | 73.0 | +0.052 [+0.009, +0.097] |
| Ensemble (LR stacker) | ensemble | 0.887 | 0.818 | 0.953 | 0.903 | 0.880 | 80.0 | +0.040 [+0.007, +0.073] |
| Random Forest on features v2 | classical | 0.885 | 0.779 | 0.929 | 0.725 | 0.947 | 67.7 | +0.078 [+0.028, +0.132] |
| Extra Trees on features v2 | classical | 0.881 | 0.772 | 0.934 | 0.715 | 0.947 | 66.8 | +0.085 [+0.033, +0.143] |
| Logistic regression on features v2 | classical | 0.873 | 0.769 | 0.937 | 0.755 | 0.919 | 68.4 | +0.088 [+0.031, +0.145] |
| CNN bidirectional cross-attention (deep) | deep | 0.857 | 0.775 | 0.933 | 0.878 | 0.848 | 75.4 | +0.082 [+0.051, +0.115] |
| TCN (deep) | deep | 0.856 | 0.760 | 0.919 | 0.813 | 0.873 | 70.9 | +0.097 [+0.065, +0.132] |
| ResNet1D (deep) | deep | 0.852 | 0.750 | 0.915 | 0.792 | 0.876 | 69.2 | +0.108 [+0.073, +0.143] |
| CNN cross-attention fusion (proposed) | deep | 0.844 | 0.746 | 0.915 | 0.813 | 0.857 | 69.9 | +0.111 [+0.073, +0.151] |
| CNN concat fusion (deep) | deep | 0.837 | 0.744 | 0.909 | 0.838 | 0.837 | 70.9 | +0.114 [+0.078, +0.152] |
| HistGB on features v1 | classical | 0.829 | 0.735 | 0.914 | 0.842 | 0.824 | 70.4 | +0.123 [+0.072, +0.174] |
