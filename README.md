# FD-PINNs for High Reynolds Number Navier-Stokes Equations

## Project Overview

This project implements a Physics-Informed Neural Network (PINN) enhanced with finite difference methods to solve the 2D Navier-Stokes equations (cavity flow) at **high Reynolds numbers**.

Standard PINNs use automatic differentiation to compute derivatives, which often leads to instability and accuracy issues in high Reynolds number regimes. By replacing automatic differentiation with **finite difference approximations (central differences)**, this FD-PINN approach improves stability and accuracy, effectively filling the gap where traditional PINNs struggle.

## Key Features

- Numerical computation of derivatives via central finite differences instead of autograd.
- Hard enforcement of boundary conditions for velocity fields.
- Training with a combination of Adam and L-BFGS optimizers.
- Designed to handle complex flow with high Reynolds numbers, where vanilla PINNs typically fail.

## Method Summary

1. Defines a 2D domain with a uniform grid (dx = dy = 0.01).
2. A neural network predicts velocity components (u, v) and pressure (p) at all points.
3. Partial derivatives of u, v, p with respect to x and y are computed using central difference formulas.
4. Hard boundary conditions are applied directly to velocity at domain edges.
5. The loss function is constructed from residuals of continuity and momentum equations of Navier-Stokes.
6. Model training first uses Adam optimizer, followed by L-BFGS for fine-tuning.
7. Post-training, velocity and pressure fields as well as streamlines are visualized.


## Result


