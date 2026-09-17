---
license: mit
configs:
- config_name: default
  data_files:
  - split: questions_with_answers
    path: questions_w_answers.jsonl
  - split: questions
    path: questions.jsonl
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- medical
pretty_name: K-QA
size_categories:
- 1K<n<10K
---
## K-QA

We are excited to announce the release of K-QA!

This benchmark consists of two parts: a medium-scale corpus of diverse real-world medical inquiries written by patients on K Health (an AI-driven clinical platform), and a subset of carefully crafted answers annotated by a team of in-house medical experts.

The dataset comprises 201 questions and answers, containing over 1,589 ground-truth statements.
Additionally, we provide 1,212 authentic patient questions.


For further details, refer to the [paper](https://arxiv.org/abs/2401.14493).

The recommended evaluation scheme for fine-grained evaluation can be found [here](https://github.com/Itaymanes/K-QA)


#### Cite Us
```markdown
@misc{manes2024kqa,
      title={K-QA: A Real-World Medical Q&A Benchmark}, 
      author={Itay Manes and Naama Ronn and David Cohen and Ran Ilan Ber and Zehavi Horowitz-Kugler and Gabriel Stanovsky},
      year={2024},
      eprint={2401.14493},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
