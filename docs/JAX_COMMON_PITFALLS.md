# JAX/MJX Coding Guidelines & Common Pitfalls

This document summarizes common errors and best practices when working with JAX and Mujoco MJX to avoid compilation and runtime issues.

## 1. Array Updates (Immutable Arrays)

**Incorrect (AttributeError):**
JAX arrays are immutable. You cannot call methods like `.add()` directly on an array instance unless it is part of the fluent `.at` interface.
```python
x = jnp.zeros(10)
x = x.add(1.0)  # AttributeError: 'DeviceArray' object has no attribute 'add'
x[0] = 1.0      # TypeError: 'DeviceArray' object does not support item assignment
```

**Correct:**
Use standard operators or the `.at[].method()` syntax.
```python
# Simple addition (element-wise)
x = x + 1.0

# Update specific indices
x = x.at[0].set(1.0)
x = x.at[idx].add(5.0)
```

## 2. In-Place Modifications

**Incorrect:**
```python
def bad_mod(x):
    x += 1  # In non-jit code, this rebinds x. In loops/transforms, it might be ambiguous.
    x[0] = 5 # Error
    return x
```

**Correct:**
Always return the new array state.
```python
def good_mod(x):
    x = x + 1
    x = x.at[0].set(5)
    return x
```

## 3. Control Flow (Conditionals)

**Incorrect (TracerBoolConversionError):**
Python `if/else` checks logic on the *concrete* value. Inside `@jit`, values are Tracers (symbolic), so you cannot check their value at tracing time.
```python
@jit
def step(x):
    if x > 0:  # Error: Abstract tracer value encountered
        return x
    else:
        return 0
```

**Correct:**
Use `jnp.where` for element-wise conditions or `jax.lax.cond` for branching logic.
```python
@jit
def step(x):
    return jnp.where(x > 0, x, 0)
```

## 4. Loops

**Incorrect:**
Standard Python loops (`for i in range(n)`) are unrolled during JIT compilation. If `n` is large or dynamic, this causes extremely slow compilation or hangs.

**Correct:**
Use `jax.lax.scan` (carry state) or `jax.lax.fori_loop` (side effects on buffers, usually less idiomatic in functional JAX).
```python
# Scan pattern
def body_fun(carry, x):
    new_carry = carry + x
    return new_carry, new_carry

final, stack = jax.lax.scan(body_fun, init=0, xs=jnp.arange(1000))
```

## 5. Random Numbers

**Incorrect:**
Numpy's global state (`np.random.seed`) does not work.

**Correct:**
Explicitly pass PRNGKeys.
```python
key = random.PRNGKey(42)
key, subkey = random.split(key)
val = random.uniform(subkey, shape=(10,))
```

## 6. Type Consistency

**Issue:**
Mixing `np.array` (float64 default) and `jnp.array` (float32 default often) can cause excessive recompilation or type promotion warnings.

**Best Practice:**
Be explicit with dtypes, especially when initializing from Mujoco (which is float64).
```python
qpos = jnp.array(m.qpos0, dtype=jnp.float32)
```
