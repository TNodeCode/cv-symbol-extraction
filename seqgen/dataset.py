import os
import pickle


def get_class_samples(img_dir):
    if os.path.exists("class_samples.pkl"):
        with open("class_samples.pkl", "rb") as f:
            class_samples = pickle.load(f)
    else:
        classes = os.listdir(img_dir)
        class_samples = {}
        for c in classes:
            class_samples.update({c: []})
            for f in os.listdir(f"{img_dir}/{c}"):
                if os.path.isfile(f"{img_dir}/{c}/{f}"):
                    class_samples[c].append(f)

        with open("class_samples.pkl", "wb") as f:
            pickle.dump(class_samples, f)
    return class_samples