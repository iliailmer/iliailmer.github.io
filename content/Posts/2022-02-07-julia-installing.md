---
Title: Installing Julia
date: 2021-12-6
status: hidden
tags: Julia
---

# Introduction

In this quick tutorial, I will share how to install Julia language on a Mac computer, setup symlinks to run it from command line and then will go through a couple of problems in probability and their Julia language simulations.



## Installing Julia on macos

We are going to install a Long-Term-Support version of Julia, specifically, the latest iteration of Julia 1.6.

To this end, [follow this link](https://julialang-s3.julialang.org/bin/mac/x64/1.6/julia-1.6.5-mac64.dmg) to download the latest release (Intel-based). __Note__: if you would like to install the latest Julia version which is, as of writing this, Julia-1.7.1, you can follow [this link for Intel-based macs](https://julialang-s3.julialang.org/bin/mac/x64/1.7/julia-1.7.1-mac64.dmg) and [this link for M1-based macs](https://julialang-s3.julialang.org/bin/mac/aarch64/1.7/julia-1.7.1-macaarch64.dmg). The last one is experimental and not all packages may work with it!

Once your download is done, run the .dmg file by clicking on it. The Julia-1.6 application file will pop up. Add it to your Applications folder. To be able to call `julia` from command line, open up your terminal and run the following:

```zsh
ln -s /Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia  /usr/local/bin/julia
```

If you downloaded Julia-1.7, you will have to run:
```zsh
ln -s /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia  /usr/local/bin/julia
```

Then run the following command to make apply changes
```
exec $SHELL
```

and, finally, run
```zsh
julia
```
to see something like this:

```zsh
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.6.5 (2021-12-19)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> 
```

If you see this, then you are done!
