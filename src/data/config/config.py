data_config = {

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

    "vqav2": {
        "hf_name": "lmms-lab/VQAv2",
        "split": "validation",
        "image_column": "image",
        "question_column": "question",
        "answer_column": "answers",
        "task": "vqav2",
    },
}