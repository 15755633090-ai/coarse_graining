# Rescue screening (development only until selection is frozen)

| variant | dev mean ratio t200/500 | dev t500 ratio | dev t500 AP | dev t1000 ratio |
|---|---:|---:|---:|---:|
| epsilon_terminal_original_lr | 46.4394 | 88.1372 | 0.1336 | 179074.8303 |
| epsilon_terminal | 1.2935 | 1.2900 | 0.1359 | 1409.1670 |
| epsilon_random | 1.3615 | 1.4442 | 0.1378 | 1611.6953 |
| preconditioned_terminal | 0.8543 | 0.8520 | 0.3401 | 0.8474 |
| preconditioned_random | 0.7407 | 0.8248 | 0.3786 | 0.8381 |
| preconditioned_random_zero | 0.7378 | 0.8265 | 0.3783 | 0.8334 |
| preconditioned_t200 | 0.8461 | 1.0803 | 0.2704 | 1.3948 |
| preconditioned_t500 | 0.7925 | 0.8181 | 0.3965 | 0.8769 |
| preconditioned_curriculum | 0.7735 | 0.8541 | 0.3565 | 0.8623 |
| velocity_random | 0.8003 | 0.8697 | 0.3323 | 0.8671 |
| bounded_random | 0.7169 | 0.8225 | 0.3838 | 0.8531 |
| bounded_density | 0.7347 | 0.8332 | 0.3738 | 0.8634 |
| bounded_density_brics | 0.6502 | 0.6913 | 0.5364 | 0.7346 |
| bounded_density_gate | 0.5970 | 0.6713 | 0.5632 | 0.7165 |
| bounded_density_gate_balanced | 1.4350 | 1.4811 | 0.4944 | 1.5675 |
| bounded_density_gate_degree | 0.6369 | 0.6858 | 0.5395 | 0.7346 |

## Confirmation (not used to pick or retune candidates)

| variant | seed | confirm mean ratio t200/500 | confirm t500 AP | confirm t1000 ratio |
|---|---:|---:|---:|---:|
| epsilon_terminal | 7 | 1.3238 | 0.1296 | 1467.5389 |
| epsilon_terminal | 1 | 1.3537 | 0.1301 | 304.1318 |
| epsilon_terminal | 2 | 1.2408 | 0.1310 | 241.5040 |
| epsilon_random | 7 | 1.3968 | 0.1326 | 1667.8662 |
| epsilon_random | 1 | 1.2567 | 0.1369 | 78.2323 |
| epsilon_random | 2 | 2.4457 | 0.1358 | 2813.7023 |
| bounded_density_gate | 7 | 0.6122 | 0.5405 | 0.7262 |
| bounded_density_gate | 1 | 0.6126 | 0.5584 | 0.7062 |
| bounded_density_gate | 2 | 0.6175 | 0.5696 | 0.6901 |
