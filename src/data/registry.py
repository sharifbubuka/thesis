REGISTRY = {

    "textvqa": {
        "hf_name": "lmms-lab/textvqa",
        "split": "train",
        "image_column": "image",
        "question_column": "question",
        "answer_column": "answers",
        "task": "textvqa",
    },

    "gqa": {
        "hf_name": "vikhyatk/gqa",
        "split": "train_balanced",
        "image_column": "image",
        "question_column": "question",
        "answer_column": "answer",
        "task": "gqa",
    },

    "vizwiz": {
        "hf_name": "lmms-lab/VizWiz-VQA",
        "split": "val",
        "image_column": "image",
        "question_column": "question",
        "answer_column": "answers",
        "task": "vizwiz",
    },
}