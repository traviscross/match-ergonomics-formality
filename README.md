# Formality for Match Ergonomics

This tool formalizes the operation of match ergonomics and implements
a number of different proposed rules.

# Usage

This tool has three main features:

- For a given (simplified) let statement, show the resulting type of
binding under a number of different proposed rulesets.

- For a given set of rules, explain the operation of match ergonomics
step by step.

- For a given set of rules, generate a state transition diagram
showing how the default binding mode (DBM) is allowed to transition
from one state to another based on the pattern and the scrutinee.

To start, type a simple let statement, e.g.:

```
> let [[x]] = &[&mut [T]];
```

The tool assumes that all lowercase letters are binding names and all
uppercase letters are unit types (that implement `Copy`).  It supports
only slices, and it does not support multiple bindings (or commas
generally).

When you do this, we'll print out the type of the resulting binding
according to a number of different rulesets.

If you want to see this explained, step by step, according to the
current stable rules, type:

```
> explain
```

Then just press return to proceed through the steps.

If you want to see things explained under a different set of rules,
you can `set` and `unset` rules.

The orthogonal rules are:

  - `rule1`
  - `rule2`
  - `rule3`
  - `rule4`
  - `rule4_ext`
  - `rule5`
  - `rule3_ext1`
  - `rule3_lazy`
  - `rule4_early`
  - `spin_rule`

And we provide these aliases:

  - `stable`: Unset all rules.
  - `proposed` / `rfc`: Set rules 1-5 + `rule4_ext`.
  - `rpjohnst`: Set:
    - `rule1`
    - `rule2`
    - `rule3_lazy`
    - `rule4_early`
    - `spin`
  - `waffle`: Set:
    - `rule1`
    - `rule3`
    - `rule3_ext1`
    - `rule4_early`

So we can write, e.g.:

```
> set proposed
> unset rule5
```

...if we want to explore how things work under the proposal without
Rule 5.

Once we've set the rules to our liking, we can see a graph of the
state transition diagram with:

```
> graph
```

This will print a link to a mermaid diagram.

To see your current settings, including all rules applied, type:

```
> show
```

Press ctrl-d to exit.

# Contributing

This tool is still under heavily development by the author and is not
yet seeking outside contributions.  Expect force pushes on the master
branch.

# License

This project is distributed under the terms of both the Apache License
(Version 2.0) and the MIT License.

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT)
for details.
