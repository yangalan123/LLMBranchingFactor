import glob
import os
if __name__ == '__main__':
    models=("meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "meta-llama/Meta-Llama-3-70B")
    # output_root_dir = "response_news"
    output_root_dir = "cognac_responses_200_violating_constraint"
    constraints = [2, 3, 4, 5]
    for model in models:
        model_base = os.path.basename(model)
        constraint_dirs = [os.path.join(output_root_dir, f"application_ctrlgen_multi_constraints_{constraint}") for constraint in constraints]
        flag = True
        for constraint_i, constraint_dir in enumerate(constraint_dirs):
            # check whether update_full_spectrum exists for model
            filenames = glob.glob(os.path.join(constraint_dir, f"{model_base}_response*from_1_to_{constraints[constraint_i]}"))
            filenames = [x for x in filenames if "metadata" not in filenames]
            if not filenames or len(filenames) is None:
                flag = False
                break
            # check whether the file is bad link or if it is not a link, check if it is empty
            for filename in filenames:
                if os.path.islink(filename):
                    if not os.path.exists(filename):
                        flag = False
                        break
                else:
                    if os.path.getsize(filename) == 0:
                        flag = False
                        break
        if flag:
            # print(f"Model {model_base} has update_full_spectrum for all constraints")
            print(f"Model {model_base} has tags for all constraints")

