# Run 12 · deepseek-after-fix

whole page : 70/85 = 82.4%
body only  : 46/53 = 86.8%   <- the rule reads this

| page | label | body | read | missing | invented | critical |
|---|---|---:|---:|---|---|---|
| p020 | ٢٢ | 5/6 | 6 | 310 | 31 | ٣١٠ FAIL |
| p035 | ٣٧ | 2/7 | 7 | 125 126 127 128 72 | 135 136 137 138 27 | ٧٢ FAIL ١٨ok |
| p062 | ٦٤ | 3/3 | 3 | - | - | ٢٣٣ok |
| p078 | ٨٠ | 5/5 | 5 | - | - | ٣٠٣ok |
| p094 | ٩٦ | 8/8 | 8 | - | - | ٣٥٧ok ١٢٤ok |
| p108 | ١١٠ | 11/12 | 12 | 414 | 41 | ١ok ٤١٤ FAIL |
| p121 | ١٢٣ | 6/6 | 6 | - | - | ١٤٨ok |
| p134 | ١٣٦ | 6/6 | 6 | - | - | ٥٣٧ok |

## the rule, as written before any of this ran

- body recall        : 86.8%
- critical failures  : 3  ['p020:٣١٠', 'p035:٧٢', 'p108:٤١٤']
- invented (spurious): 7  ['p020:31', 'p035:135', 'p035:136', 'p035:137', 'p035:138', 'p035:27', 'p108:41']

**3. REJECT -- ADR-029 stands unchanged.**

## whole-page view (masking check)

- recall 82.4%
- invented: ['p020:21', 'p020:31', 'p020:54', 'p035:135', 'p035:136', 'p035:137', 'p035:138', 'p035:21', 'p035:27', 'p035:54', 'p094:5', 'p108:21', 'p108:41', 'p121:21', 'p121:54']
- critical failures: ['p020:٣١٠', 'p035:٧٢', 'p108:٤١٤']
