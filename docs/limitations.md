# Limitations

PI-GINOT performs numerical verification against finite element method (FEM)
solutions. That is not experimental validation, and it does not show that predictions
agree with measurements from physical specimens.

Verification for a bundle establishes nothing about geometries outside that bundle's
declared input envelope. Inputs outside the envelope must not be treated as covered by
the bundle's results.

Finite verification sampling is evidence about the cases evaluated. It is not a
mathematical proof of a uniform error bound over every admissible input.

At the current S0 stage, PI-GINOT contains only repository scaffolding and version
reporting. It does not yet implement FEM, neural-operator training, verification, or
inference.
