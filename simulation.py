import numpy as np


def run_simulation_expreiment(signal, regressors, N=250, N_trials=20, sd=0.1):
    trials = run_trials(signal, regressors, N, N_trials, sd)
    
    def make_mean_errors(trials):
        train_mean_errors = trials[0].mean(axis=0)
        test_mean_errors = trials[1].mean(axis=0)
        return train_mean_errors, test_mean_errors
    
    def make_standard_errors(trials):
        train_standard_errors = trials[0].std(axis=0)
        test_standard_errors = trials[1].std(axis=0)
        return train_standard_errors, test_standard_errors
    
    mean_errors = make_mean_errors(trials)
    standard_errors = make_standard_errors(trials)
    return mean_errors, standard_errors

def plot_simulation_expreiment(ax, degrees_of_freedom, mean_errors, std_errors):
    train_me, test_me = mean_errors
    train_se, test_se = std_errors
    train_top_band, train_bottom_band = train_me + train_se, train_me - train_se
    test_top_band, test_bottom_band = test_me + test_se, test_me - test_se

    ax.plot(degrees_of_freedom, train_me, label="Train")
    ax.fill_between(
        degrees_of_freedom, train_top_band, train_bottom_band, alpha=0.2)
    ax.plot(degrees_of_freedom, test_me, label="Test")
    ax.fill_between(
        degrees_of_freedom, test_top_band, test_bottom_band, alpha=0.2)
    ax.legend()

def make_random_train_test(signal, N=250, sd=0.1):
    x_train = np.random.uniform(0, 1, size=N)
    y_train = signal(x_train) + np.random.normal(scale=sd, size=N)
    x_test = np.random.uniform(0, 1, size=N)
    y_test = signal(x_test) + np.random.normal(scale=sd, size=N)
    return (x_train, y_train), (x_test, y_test)

def run_trials(signal, regressors, N=250, N_trials=250, sd=0.1):
    N_dof = len(regressors)
    train_errors = np.empty(shape=(N_trials, N_dof))
    test_errors = np.empty(shape=(N_trials, N_dof))
    for i in range(N_trials):
        for j, regressor in enumerate(regressors):
            (x, y), (x_test, y_test) = make_random_train_test(signal, N, sd)
            regressor.fit(x.reshape(-1, 1), y)
            y_hat_train, y_hat_test = (regressor.predict(x.reshape(-1, 1)), 
                                       regressor.predict(x_test.reshape(-1, 1)))
            train_error, test_error = (np.mean((y - y_hat_train)**2), 
                                       np.mean((y_test - y_hat_test)**2))
            train_errors[i, j] = train_error
            test_errors[i, j] = test_error
    return train_errors, test_errors
