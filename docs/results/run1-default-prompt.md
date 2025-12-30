Runs: 480 over 10 task(s), scenarios none, bug_moved; protocol: default prompt.

Source sessions: 5/10 answered within the cutoff (resumed from just before the answer); 0 sourced findings, 0 findings total; 0 working-memory snapshots; 37 provider-reasoning states.

### Scenario `none`

| arm | n | success | 95% CI | stale answers | violations | errors | resume ctx (tok) | total input (tok) | tool calls | cache hits |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| full | 30 | 90% | 74%–97% | 0 | 10 | 0 | 1088 | 4657 | 1.3 | 0.4 |
| masked | 30 | 87% | 70%–95% | 0 | 5 | 0 | 1058 | 3272 | 1.0 | 0.5 |
| self_summary | 27 | 100% | 88%–100% | 0 | 5 | 3 | 978 | 4676 | 2.1 | 1.4 |
| slim | 28 | 93% | 77%–98% | 0 | 8 | 2 | 848 | 9075 | 4.5 | 2.8 |
| slim_validated | 29 | 90% | 74%–96% | 0 | 3 | 1 | 863 | 6770 | 3.3 | 2.8 |
| slim_memory | 30 | 93% | 79%–98% | 0 | 1 | 0 | 848 | 7877 | 3.7 | 3.0 |
| slim_reasoning | 30 | 93% | 79%–98% | 0 | 9 | 0 | 999 | 9411 | 4.2 | 3.2 |
| scratch | 30 | 83% | 66%–93% | 0 | 4 | 0 | 750 | 6714 | 3.9 | 3.3 |

Per task (`none`, correct/runs):

| task | full | masked | self_summary | slim | slim_validated | slim_memory | slim_reasoning | scratch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cart_quantity | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| coupon_expiry | 3/3 | 3/3 | 3/3 | 2/2 | 2/2 | 3/3 | 3/3 | 3/3 |
| discount_inverted | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| domain_case | 0/3 | 0/3 | 3/3 | 1/3 | 0/3 | 1/3 | 1/3 | 0/3 |
| free_shipping_boundary | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| overselling | 3/3 | 3/3 | – | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| page_count_extra | 3/3 | 3/3 | 3/3 | 2/2 | 3/3 | 3/3 | 3/3 | 3/3 |
| page_zero | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 |
| pagination_overlap | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| search_case | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

### Scenario `bug_moved`

| arm | n | success | 95% CI | stale answers | violations | errors | resume ctx (tok) | total input (tok) | tool calls | cache hits |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| full | 30 | 27% | 14%–44% | 21 | 7 | 0 | 1088 | 4756 | 1.9 | 0.7 |
| masked | 30 | 27% | 14%–44% | 22 | 8 | 0 | 1058 | 4670 | 1.6 | 1.0 |
| self_summary | 30 | 33% | 19%–51% | 18 | 9 | 0 | 976 | 7590 | 3.2 | 1.7 |
| slim | 30 | 97% | 83%–99% | 1 | 4 | 0 | 848 | 7092 | 3.7 | 2.6 |
| slim_validated | 30 | 93% | 79%–98% | 2 | 7 | 0 | 863 | 6869 | 3.7 | 3.2 |
| slim_memory | 30 | 97% | 83%–99% | 1 | 3 | 0 | 848 | 6573 | 3.6 | 2.9 |
| slim_reasoning | 30 | 90% | 74%–97% | 3 | 6 | 0 | 999 | 9879 | 4.1 | 3.2 |
| scratch | 30 | 97% | 83%–99% | 1 | 3 | 0 | 750 | 5199 | 3.4 | 3.0 |

Per task (`bug_moved`, correct/runs):

| task | full | masked | self_summary | slim | slim_validated | slim_memory | slim_reasoning | scratch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cart_quantity | 0/3 | 0/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| coupon_expiry | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| discount_inverted | 1/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| domain_case | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| free_shipping_boundary | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 |
| overselling | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| page_count_extra | 1/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| page_zero | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| pagination_overlap | 0/3 | 0/3 | 1/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| search_case | 0/3 | 0/3 | 0/3 | 2/3 | 1/3 | 3/3 | 2/3 | 2/3 |

Errors: ResponseError x6

