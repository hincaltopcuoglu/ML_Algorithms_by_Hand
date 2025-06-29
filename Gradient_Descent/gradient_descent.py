import numpy as np

class GradientDescent:
    def __init__(self, loss_fn, grad_fn, lr=0.1, max_iter=1000, tol_grad=1e-6, tol_loss=1e-8):
        self.loss_fn = loss_fn
        self.grad_fn = grad_fn
        self.lr = lr
        self.max_iter = max_iter
        self.tol_grad = tol_grad
        self.tol_loss = tol_loss
        
        self.w_ = None
        self.loss_history_ = []


    def optimize(self, X, y, w_initial):
        w = w_initial.copy()
        self.loss_history_ = []
        prev_loss = np.inf

        for i in range(self.max_iter):
            gradient = self.grad_fn(X, y, w)
            w = w - self.lr * gradient

            current_loss = self.loss_fn(X, y, w)
            self.loss_history_.append(current_loss)

            # Check gradient norm tolerance
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < self.tol_grad:
                print(f"Convergence reached: Gradient norm ({grad_norm:.2e}) is below tolerance ({self.tol_grad:.2e}) at iteration {i+1}.")
                break

            # Check loss change tolerance
            loss_change = prev_loss - current_loss
            if abs(loss_change) < self.tol_loss:
                print(f"Convergence reached: Loss change ({loss_change:.2e}) is below tolerance ({self.tol_loss:.2e}) at iteration {i+1}.")
                break
            
            prev_loss = current_loss

            
        else: 
            print(f"Optimization finished after reaching max_iter ({self.max_iter}).")

        # Store final results
        # Store the final weights. This is always safe to do.
        self.w_ = w

        # THE SAFETY CHECK:
        # In Python, an empty list evaluates to `False`. A list with items evaluates to `True`.
        # So, `if self.loss_history_:` is a simple way to ask "Did the loop run at least once?"
        if self.loss_history_:
            # This code block only runs IF the list is NOT empty (i.e., max_iter > 0).
            # It's now safe to get the last item.
            print(f"Final Loss: {self.loss_history_[-1]:.6f}")
        else:
            # This code block only runs IF the list IS empty (i.e., max_iter = 0).
            # We can't print a "final" loss, so we print a helpful message instead.
            # We can even calculate the initial loss for the user's benefit.
            initial_loss = self.loss_fn(X, y, w)
            print(f"Optimization did not run (max_iter=0). Initial Loss: {initial_loss:.6f}")

        # The return statement is safe and runs in either case.
        return self

