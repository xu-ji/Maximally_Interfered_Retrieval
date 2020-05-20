import pickle
import os
# loop through pickle files and print best

for dataset in ["split_cifar10", "miniimagenet"]:
    for strategy in ["rand", "MIR"]:

        best_run = {"valid": -1, "test": -1}
        best_res = {"valid": None, "test": None}
        for run in range(100):
            results_p = "Results/%s_hparam_search_%s/%d.pickle" % (dataset, strategy, run)
            if not os.path.exists(results_p):
                break

            with open(results_p, "rb") as f:
                results = pickle.load(f)

                print(results)

                for mode in ["valid", "test"]:
                    if best_res[mode] is None or best_res[mode]["accuracy"] < results[mode]["accuracy"]:
                        best_run[mode] = run
                        best_res[mode] = results

        print("Best results for %s %s:" % (dataset, strategy))
        print(best_run)
        print(best_res)
