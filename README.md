## NetJIT

NetJIT is used to anticipate network traffic of distributed machine learning in a program-behavior-aware way, by utilizing the runtime program/code analysis ability provided by JIT analysis.

### Usage:

To enable NetJIT on any distributed python program training a ML model, one only has to:
* Include NetJIT to the Python Path
* Import NetJIT in the python file
* Annotate the entrypoint function (the main function in most cases) with parameters to control the behavior of NetJIT
* Enable the rules corresponding to the ML framework (PyTorch for example)

It should be similar to the following code:

```python
import predictor


@predictor.entrypoint(max_analyze_depth=150, min_tracing_possibility=0.5, min_report_time=1, max_report_time=3.0, max_tracing_time=3.0)
@predictor.rules.enable_common_rules
@predictor.rules.enable_pytorch_rules
def main():
    ...
```

By default, NetJIT will write all its prediction results to `report.txt` in the directory where the python script containing the main function is located. To change the behavior or to customize the prediction outputs, users can provide their own reporters.

### Credits

This repository contains codes from the following repositories as the examples of NetJIT:

* [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) (BERT) (Apache-2)
* [PyTorch Examples](https://github.com/pytorch/examples) (GPT-1, GPT-2) (BSD)
* [LambdaLabsML Examples](https://github.com/LambdaLabsML/examples) (ResNet-152) (MIT)