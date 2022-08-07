# Minitorch Self-Study Guide (SAIA)

_By Gabriel Mukobi (Stanford AI Alignment)_

_Created: **Aug 6, 2022**. Updated: **Aug 7, 2022**_

# Intro

## What is this?

- [Minitorch](https://minitorch.github.io/): Reimplement the core functionality of the popular PyTorch machine learning library in Python.
- Similar things to parts of this (particularly Autodiff) are done in Redwood Research’s [MLAB](https://www.alignmentforum.org/posts/3ouxBRRzjxarTukMW/apply-to-the-second-iteration-of-the-ml-for-alignment) curriculum (“Implement a simple clone of some of Pytorch, with particular focus on the implementation of backpropagation”).
- Minitorch is a bit outdated and some of the instructions are unclear, so this guide should help speed you past the annoying stuff so you can efficiently complete the actual learning.

## Prerequisite Knowledge

- Some solid programming skills (e.g. CS 106B or equivalent)

- Basic machine learning skills (have trained any neural network with PyTorch/TensorFlow/Scikit-Learn/etc.). See TODO if not

  - If you have a good amount of time (~20 hours, might do this instead of Minitorch) and want to form a solid understanding of machine learning as a field (deep learning is a subset of machine learning is a subset of artificial intelligence), you should do [the FastAI course](https://course.fast.ai/). If you have less time, are solid with NumPy, want to learn more after the FastAI course, or have a general idea of how some ML works (note that we'll read and learn a lot about ML training in Minitorch), you might do [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
  - Having ML/PyTorch experience is only really useful to Minitorch in that you'll have a better idea of what the functions you're building are supposed to do, but it will be pretty important later once we get to implementing transformers and doing AI safety paper replications.

## Time Commitment

- Very roughly, 5 hours of programming + 1 hour of discussion per week over 4 or 5 weeks, making 20-30 hours total.

# [Module 0: Fundamentals](https://minitorch.github.io/module0.html)

> This introductory module is focused on introducing several core technologies used for testing and debugging in future modules, and also includes some basic mathematical foundations. In this module, you will start to build up some of the infrastructure for MiniTorch.

## 0.0: Get started on Minitorch

1. Open <https://minitorch.github.io/>
2. Read through the **Setup**, the **ML Primer**, and the guides for **Fundamentals** (guides are listed in the sidebar of the webpage on a computer).
3. Do the tasks for Fundamentals by forking and cloning or downloading <https://github.com/minitorch/Module-0> (I clicked the blue "Use this template" button and named my repo “Minitorch-Module-0”). Module 0 is pretty straightforward, but if something is weird or you are having trouble getting set up, say so and we'll all help each other!
4. Minitorch is a bit outdated, and I recommend removing all the version numbers in**requirements.txt** and **requirements.extra.txt** (e.g. change the line "numpy == 1.19.1" to just "numpy") and adding "torch" as a new line in requirements.txt before installing to get the version numbers to be compatible. This works as of Aug 6, 2022, but if you’re from the future and it doesn’t work, maybe try changing the hardcoded version numbers in those files to newer but still compatible versions rather than removing them completely.

## 0.1-0.4: Common issues

5. No file app.py in Task 4: You have to cd into the**Project** folder first before running streamlit, the docs don't tell you this.
6. numba 0.55 required numpy &lt; 1.22, but torch 1.11 requires numpy >= 1.22: This happens when you use Python 3.10 or above. Install and use Python 3.9 instead and this should go away.
7. Other weird things: Make sure you forked or downloaded **minitorch/Module-0** that I linked above and _not_ the full minitorch/minitorch repository.

# [Module 1: Autodiff](https://minitorch.github.io/module1.html)

> This module shows how to build the first version of MiniTorch (mini-MiniTorch?) using only Scalar values. This covers key aspects of auto-differentiation: the key technique in the system. Then you will use your code to train a preliminary model.

## 1.1: Numerical Derivatives

1. Be sure to template off of/fork/clone <https://github.com/minitorch/Module-1>, _not_ the full minitorch/minitorch repository.
2. You can reuse your same virtual environment from Module-0 for this one, but be sure to remove the version numbers from Module-1's requirements.txt and requirements.extra.txt and run the same install commands. You will do this for each module because some modules have different required packages that won't be in your virtual environment from the previous module.

_Tip: I created scripts to automate the virtual environment activation and running the installation commands. Assuming you’re using Windows (for Powershell) and a Python virtual environment called venv installed in the folder above your Module 1, these scripts look like:_

```powershell
..\\venv\\Scripts\\Activate.ps1
```

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements.extra.txt
python -m pip install -Ue .
```

_You can of course modify these scripts for your particular environment._

3. Also for each module, bring over the files you edited from all the previous modules. That means you should copy over **minitorch/module.py**, **minitorch/operators.py**, and optionally (if you want the tests from the previous module) **tests/test_module.py** and **tests/test_operators.py**. For Module-1, for example, do _not_ bring over minitorch/testing.py since we didn't edit it in any of the previous modules (just Module-0).

## 1.2: Scalars

1. Where it says "Complete the following functions in minitorch/scalar_functions.py," scalar_functions.py doesn't actually exist. The functions you're supposed to implement are in **scalar.py**.
1. If you're failing Scalar tests with `Expected return type <class 'float'> got <class 'int'>`, make sure you're writing things like `0.0` or `1.0` instead of `0` or `1` in your **operators.py** (or wherever you're writing the base Python operations). If you write something like `max(x, 0)`, Python will interpret the return type as an `int`.

## 1.3: Chain Rule

1. Similarly, `FunctionBase.chain_rule` is actually in `autodiff.py`, not `Scalar.chain_rule` in `scalar.py`

## 1.4: Backpropagation

1. 1.4 is a bitch, but you'll learn a lot here.
1. Topological sort is confusing as they present it, read the first couple of sections of <https://en.wikipedia.org/wiki/Topological_sorting> then implement "Depth-First-Search" (note we don't need the temporary markers from the Wikipedia algorithm).
1. `variable.history.inputs` will give you the variables that fed into the creation of this variable.
1. There aren't any tests for `topological_sort`, so consider writing your own tests or debugging a call to it to see if it makes sense.
1. Use `variable.unique_id` to add a hashable type (`unique_id` is a `string`, `Variable` is unhashable) to sets or dictionary keys.
1. You also have to implement `History.backprop_step`, they don't tell you that. You'll need to call `self.last_fn.chain_rule`, loop through the input variables and corresponding derivatives, add either the derivative value or `0.0` if the variable `is_constant` to an output list, and return the output list converted to a `tuple`.
1. The reason why we need the topologically sorted ordering will be most clear once you implement backpropagate which is unfortunate.
1. The derivative of `sigmoid(x)` is `sigmoid(x) * (1 - sigmoid(x))`.
1. The derivative of less than and of equals are both `0.0` (think about why this is). Make sure you return the same number of elements in `backward` as the number of inputs, so since both LT and EQ take two variables as input (a and b), you should `return 0.0, 0.0` in backward().
1. An annoying thing is that Python functions which return tuples of size 1 (e.g. `def fun(): return (5)`) will not return a tuple (e.g. `(5)`), they'll return the inner value (e.g the `int` `5`) (I don't like this because returning a tuple of size 2+ or of 0 (!) will return a type of tuple). When you call `backprop_step` in `backpropagate`, wrap the result in `wrap_tuple()` in order to convert single `float` values returned to a `tuple` of `float`s.

## 1.5: Training

1. You'll be modifying **project/run_scalar.py** and running that Python script to test it. When finished, you should be able to run that script and get 50 correct on the simple dataset and >45 correct on the XOR dataset (46 is the best I got, I could be wrong or not tested enough and you could get better though). Note the PTS parameter indicates the total number of points in the dataset, 50 by default.
1. The input size for each dataset is 2, and the output size for each dataset is 1 (they're all functions from 2 numbers to 1 number, see **minitorch/datasets.py**).
1. The `hidden_layers` parameter should probably be called `hidden_size`. With the way the starter code is structured, you'll always have three layers in your `Network` class, but you'll change the hidden dimension connecting the layers from 2 for the Simple dataset to 10 for the Xor dataset.
1. If your code runs but the printed-out correct numbers don't change (e.g. stick around 26) or quickly drop to zero, try running the training again. Neural networks can be very sensitive to the random initialization state of their parameters.

# [Module 2: Tensors](https://minitorch.github.io/module2.html)

> Tensors group together many repeated operations to save Python overhead and to pass off grouped operations to faster implementations.

If you want even more practice with tensors through a cool interactive visualization tool, see <https://github.com/srush/tensor-Puzzles>.

## 2.1: Tensor Data - Indexing

1. As before, remember to remove the version numbers from requirements.txt and requirements.extra.txt as well as to copy over the minitorch files from previous modules (**minitorch/autodiff.py**, **minitorch/module.py**, **minitorch/operators.py**, and **minitorch/scalar.py**)
1. Indexing is hard (at least for me). For `to_index`, consider using `strides_from_shape()` to get a tuple of strides, then using division and truncation to integers. Note: If you do it this way, you'll have to rewrite it without `strides_from_shape()` in Module 3 due to a Numba compilation error.

## 2.2: Tensor Broadcasting

1. They don't provide a test for broadcast_index (and it's not used until later in this module), here's one I quickly wrote that looks right:

```python
@pytest.mark.task2_2
def test_broadcast_index():
    c = [0]
    minitorch.broadcast_index((2, 3), (5, 5), (1,), c)
    assert c == [0]

    c = [0]
    minitorch.broadcast_index((2, 3), (5, 5), (5,), c)
    assert c == [3]

    c = [0, 0]
    minitorch.broadcast_index((1, 2, 3), (1, 5, 5), (5, 5), c)
    assert c == [2, 3]

    c = [0, 0, 0, 0]
    minitorch.broadcast_index((4, 3, 2, 1), (5, 5, 4, 4), (5, 5, 1, 1), c)
    assert c == [4, 3, 0, 0]
```

## 2.3: Tensor Operations

1. The TODOs in **tensor_ops.py** are mislabeled, as map/zip/reduce are for 2.3, not 2.2.
1. The start of make_tensor_backend instantiates some mapped/zipped functions for you.
1. You'll probably need to call broadcast_index broadcasting from the larger output shape to the smaller input shape(s).

## 2.4: Gradients and Autograd

1. If you're failing `test_grad_size` among other things, you may need to edit your **autodiff.py**. You will need to call expand somewhere in it (e.g. `derivative = input.expand(derivative)`) in order to make sure the output of `backward` is always the same dimensions as the input of `forward`. You can check out https://github.com/lvsizhe/Module-2/blob/master/minitorch/autodiff.py for a reference implementation if you're stuck.
1. `Permute.forward` requires accessing the inner `TensorData` object, calling its `permute` function with the unzipped `*order`, and constructing a new `Tensor` object to return with `a._new()`. `Permute.backward` requires something similar, but permuting `grad_output` in the reverse order as the input was permuted (you'll have to be a bit clever).

## 2.5: Training

1. The instruction webpage doesn't say it, but you have to edit **project/run_tensor.py** to implement the model's `forward` call and a `Linear` layer's `forward` using tensors, then run **project/run_tensor.py** to test it.
1. You can copy `Network.forward()` from **project/run_torch.py**.
1. `Linear.forward()` is trickier, as you have to account for batching and use `view()` a lot along with multiplying and summing. There are vague instructions in the Broadcasting web guide, or you can just see https://github.com/lvsizhe/Module-2/blob/master/project/run_tensor.py for a reference implementation if you're stuck.
1. I had a bug in my `sigmoid` tensor function that wasn't being caught in the unit tests but caused training to fail (I forgot to multiply by `grad_output`), so keep in mind there may be hidden bugs in your code and test small things.

# [Module 3: Efficiency](https://minitorch.github.io/module3.html)

> This module is focused on taking advantage of tensors to write fast code, first on standard CPUs and then using GPUs.

See https://github.com/srush/GPU-Puzzles for help writing CUDA kernels.

## 3.1: Parallelization

1. I think "All indices use numpy buffers" means if you are creating local variables to represent indices into tensors, they should be of type `np.ndarray`, not a vanilla Python list (because Numba can't optimize Python lists like it can Numpy arrays). Really, it should say "use Numpy arrays" instead of "buffers".
1. I think "When out and in are stride-aligned, avoid indexing" means that if the strides and the shapes of out and in are the same, you can just operate on the linear data storage rather than doing any indexing. You must check both stride and shape (e.g. a shape 2,2 tensor can have strides 1,2 or 2,1) as well as the number of dimensions (i.e. `len(out_shape)`).
1. For `reduce`, I'm not sure how "Inner-loop should not call any functions" is possible, and all the implementations in other forks I looked at seem to call functions within their inner loop.
1. If you want to disable Numba JIT compilation when running PyTest so you can debug your code or get better error messages, the easiest way I found was to create a pytest.ini file in the root project directory and add to it:

```ini
[pytest]
env =
    NUMBA_DISABLE_JIT=1
```

1. If you're getting errors about `strides_from_shape` (or potentially other functions) not being found, it's because the starter code isn't set up to JIT compile them. In my case, I was calling `strides_from_shape` within my implementation of `to_index` (which is JIT compiled), so rewriting `to_index` so that it doesn't use `strides_from_shape` fixes this.
1. `Untyped global name` errors where the name is an operator: Numba only wants you to use a limited subset of Python, it doesn't like it when you call a bunch of non-builtin or non-Numpy functions. So if in your **operators.py** you implemented functions like `neg` as `return mul(-1.0, x)`, just change it to `return -x` (and similar usages of `lt`, `neg`, etc.).
1. Speaking of which, because we're dealing with Numpy arrays, the `==` operator returns an array of `bool`s indicating whether each value is the same or not, so we need to use `.all()` to check that they are all the same.

## 3.2: Matrix Multiplication

1. It's also not clear to me how to satisfy "No index buffers or function calls" without just inlining all your `tensor_data` functions which Numba should be able to JIT for you.
1. My `broadcast_index` from before had some asserts in it that I used to be more careful about broadcasting when I was writing it (e.g. you technically shouldn't be able to broadcast a shape (1,2,4) tensor to shape (1,2,3)). However, it's useful to remove those asserts to bend the rules of broadcasting a little for the purposes of easily generating indices for matrix multiplying.
1. project/parallel_test.py does not exist, so I think they must have meant **project/parallel_check.py** again.

## 3.3: CUDA Operations

## 3.4: CUDA Matrix Multiplication

## 3.5: Training

TODO 

# [Module 4: Networks](https://minitorch.github.io/module4.html)

TODO
