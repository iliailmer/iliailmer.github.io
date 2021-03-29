---
Title: How I had to translate Matlab code into Maple
slug: matlab-2-maple
date: 2020-08-18
Tags: python,regular expressions,matlab,maple
---

In this short post, I wanted to point out one interesting application of regular expressions I had to work on for my PhD research project. The code was meant as a technical tool to help tranlate some ordingary differential equation models from numerical (Matlab) to symbolic (Maple) code.

## The original code

The original \*.m files were pulled from [this](https://www.ebi.ac.uk/biomodels/) webpage using `wget` and their own API. Ordingary differential equation (ODE) models in those files contain a special function called `xdot`. It returns an array of `n` elements, which form right-hand side of an ODE.

The function is usually defined in the form

```m
function xdot=f(x,t)
    % define parameters as
    C = 1.0;
    % ...
    xdot = zeros(..., ...);
    % define each xdot(i) separately
end
```

What I wanted to see in the maple code was the following:

```py
#  define parameters as symbolic constants
C := C:

#  define system of odes as array

sigma := [
    diff(x1(t), t) = <...>,
    <...>
    diff(xn(t), t) = <...>
]:
```

The best way to solve it I saw was to use regular expressions.

## Step by step

Firstly, I had to get rid of Matlab comment and turn them into Maple compatible ones, hence the line

```py
out_program = re.findall(
                r"function xdot=f\(x,t\)(.*?)end", content, re.DOTALL
            )[0].replace("%", "#")
```

After that, we want to look at `if` statements. In Maple, conditional structure like `if` statements are written as

```pascal
if <condition> then <code> end:
```

To do that, we utilize groups: `(..)`. The regex is

```py
out_program = re.sub(
                r"if(\(\w*\))(.*;)end",
                r"if \1 then \2\nend if:",
                out_program,
                flags=re.DOTALL,
            )
```

The `if(\(\w*\))(.*;)end` regex gets the condition `\(\w*\)` and the code (.\*;) between `if` / `end` to place those between `if`, `then`, `end` in positions marked by `\1` and `\2`.

Remeber, that we do not want to have `xdot(i)=...` assigned manually anymore in Maple, we want to see Maple syntax: `diff(x(t),t) = ...`, so we do the following regex:

```py
out_program = re.sub(
                r"xdot\((\d+)\) \=( .*);", r"\ndiff(x\1(t), t) = \2,", out_program
            )
```

Again, notice the groups that capture stuff we want to preserve, namely the index of `xdot` and the right-hand sides.

Next few lines do some cosmetic work, namely, rename any `x(i)` appearance into `xi(t)`, then make all variable assingments symbolic constants (i.e. if we have `c=0` we want to have `c:=c`). Finally, if we have Matlab assignments that do not use constants (i.e. `c=0; x = c+x;`) we want to keep them (i.e. `c:=c: x:=c+x:`).

```py
out_program = re.sub(r"x\((\d+)\)", r"x\1(t)", out_program) # x(i) -> xi(t)
out_program = re.sub(r"(\w+)\=\d*\..*", r"\1:=\1:", out_program) # c=NUMBER -> c:=c:
out_program = re.sub(r"(\w+)\=(.*);", r"\1:=\2:", out_program) # Left-side = ANYTHING NOT METNIONED ABOVE -> same but with := sign
```

Finally, putting it all together we can run:

```py
import re
from glob import glob
from tqdm.auto import tqdm

files = glob("files/*/*/*.m")

for i in tqdm(range(len(files))):
    with open(files[i], "r") as f:
        try:
            content = f.read()
        except:
            print(files[i], "could not read")
        try:
            out_program = re.findall(
                r"function xdot=f\(x,t\)(.*?)end", content, re.DOTALL
            )[0].replace("%", "#")
            out_program = re.sub(
                r"if(\(\w*\))(.*;)end",
                r"if \1 then \2\nend if:",
                out_program,
                flags=re.DOTALL,
            )
            out_program = re.sub(
                r"xdot\((\d+)\) \=( .*);", r"\ndiff(x\1(t), t) = \2,", out_program
            )
            out_program = re.sub(r"(xdot=zeros.*)", r"# \1", out_program) # comment out declaration of xdot

            out_program = re.sub(r"x\((\d+)\)", r"x\1(t)", out_program)
            out_program = re.sub(r"(\w+)\=\d*\..*", r"\1:=\1:", out_program)
            out_program = re.sub(r"(\w+)\=(.*);", r"\1:=\2:", out_program)
            # cosmetic work to make maple run and not complain about trailing comma
            out_program = re.sub(
                r"(diff.*)", r"sigma := [\n\1]:", out_program, flags=re.DOTALL
            ).replace(",\n]", "\n]")

            outname = files[i].split(".")[0] + ".mpl"

            with open(outname, "w") as output:
                output.write(out_program)
            import os

            os.system("mkdir -p new_examples")
            os.system(f"cp {outname} new_examples")
        except:
            print(files[i], "Index not Found.")

```

Some stuff really specific to our project was omitted for brevity, but one can definitely add any other Maple/Matlab interaction here.
